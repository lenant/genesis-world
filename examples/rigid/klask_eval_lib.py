"""Shared evaluation helpers for the KLASK self-play example.

Both ``klask_eval.py`` (manual evaluation) and ``klask_train.py`` (periodic
in-training evaluation / best-model selection) import from here so the rollout
and scripted-opponent logic stays in a single place.
"""

import torch


def _discard_recording(cam):
    """Reset a recording camera without saving, so a failed eval leaves it clean.

    ``stop_recording()`` would auto-save a junk file, so the recorder state is reset
    directly. Best-effort: never raise from a cleanup path.
    """
    try:
        cam._recorded_imgs.clear()
        cam._recorded_t_prev = -1
        cam._in_recording = False
    except Exception:
        pass


def baseline_actions(observations, opponent):
    """Return canonical (left-frame) actions for a scripted opponent.

    ``observations`` are the per-board canonical observations of the side being
    controlled (see ``KlaskSelfPlayEnv._canonical_state``). The layout used here:
    index 0/1 = own striker x/y, 8/9 = ball x/y, 10 = ball vx.
    """
    if opponent == "passive":
        return torch.zeros((observations.shape[0], 2), dtype=observations.dtype, device=observations.device)
    if opponent == "random":
        return (
            torch.rand(
                (observations.shape[0], 2),
                dtype=observations.dtype,
                device=observations.device,
            )
            * 2.0
            - 1.0
        )

    own_x = observations[:, 0]
    own_y = observations[:, 1]
    puck_x = observations[:, 8]
    puck_y = observations[:, 9]
    puck_vx = observations[:, 10]
    defend = (puck_x < 0.05) | (puck_vx < -0.05)
    target_x = torch.where(
        defend,
        torch.clamp(puck_x - 0.16, -0.82, -0.08),
        torch.full_like(puck_x, -0.55),
    )
    target_y = torch.where(defend, puck_y, torch.clamp(puck_y * 0.75, -0.65, 0.65))
    return torch.clamp(torch.stack([target_x - own_x, target_y - own_y], dim=1) * 2.8, -1.0, 1.0)


def rollout_eval(
    env,
    policy,
    *,
    opponent="heuristic",
    num_steps,
    record=False,
    record_fps=60,
    video_path=None,
):
    """Run a deterministic-ish evaluation rollout and tally scoring statistics.

    The learned policy always controls the left slots (boards ``0..B-1``). When
    ``opponent != "self"`` the right slots (boards ``B..2B-1``) are overridden
    with a scripted controller. Boards reset in-place on a score/timeout, so a
    single rollout aggregates many completed games across all boards.

    Returns a metrics dict. When ``record`` is set and the env was built with a
    record camera, board 0 is rendered every step and saved to ``video_path``.
    """
    num_boards = env.num_boards

    can_record = record and getattr(env, "record_cam", None) is not None
    if record and not can_record:
        print("rollout_eval: record requested but env has no record camera; skipping video.")

    policy_scores = 0
    opp_scores = 0
    draws = 0
    reason_counts = {"goal": 0, "biscuits": 0, "out_of_bounds": 0}
    reward_sum = 0.0

    # rsl_rl drives the env under torch.inference_mode() during training, which makes
    # the env's persistent buffers inference tensors. Reset/step them in the same
    # mode here so in-place updates stay legal.
    recording = False
    try:
        with torch.inference_mode():
            obs_dict = env.reset()
            if can_record:
                env.record_cam.start_recording()
                recording = True

            for _ in range(num_steps):
                actions = policy(obs_dict)
                if opponent != "self":
                    right_obs = obs_dict["policy"][num_boards:]
                    actions = actions.clone()
                    actions[num_boards:] = baseline_actions(right_obs, opponent)

                obs_dict, rewards, dones, _ = env.step(actions)
                reward_sum += rewards[:num_boards].mean().item()

                if can_record:
                    env.record_cam.render()

                done_boards = dones[:num_boards].bool().nonzero(as_tuple=False).reshape((-1,))
                if len(done_boards) > 0:
                    scored_by = env.last_scored_by.detach().cpu().tolist()
                    for board_idx in done_boards.detach().cpu().tolist():
                        reason = env.last_score_reason[board_idx]
                        if reason == "none":
                            draws += 1
                            continue
                        if reason in reason_counts:
                            reason_counts[reason] += 1
                        if scored_by[board_idx] == 0:
                            policy_scores += 1
                        elif scored_by[board_idx] == 1:
                            opp_scores += 1

        # Successful rollout: finalize the recording into the requested file.
        if recording:
            if video_path is not None:
                env.record_cam.stop_recording(save_to_filename=str(video_path), fps=int(record_fps))
            else:
                env.record_cam.stop_recording()
            recording = False
    except Exception:
        # The caller swallows eval failures to keep training alive, so make sure a
        # half-finished recording does not leak frames into the next evaluation.
        if recording:
            _discard_recording(env.record_cam)
        raise

    decisive = policy_scores + opp_scores
    games = decisive + draws
    win_rate = policy_scores / decisive if decisive > 0 else 0.5
    net_score = policy_scores - opp_scores
    metrics = {
        "opponent": opponent,
        "num_steps": num_steps,
        "games": games,
        "decisive": decisive,
        "draws": draws,
        "policy_scores": policy_scores,
        "opp_scores": opp_scores,
        "win_rate": win_rate,
        "net_score": net_score,
        "net_score_per_game": (net_score / games) if games > 0 else 0.0,
        "policy_goals": reason_counts["goal"],
        "out_of_bounds": reason_counts["out_of_bounds"],
        "biscuit_scores": reason_counts["biscuits"],
        "mean_left_reward": reward_sum / max(1, num_steps),
        "video_path": str(video_path) if (can_record and video_path is not None) else None,
    }
    return metrics
