import argparse
import pickle
from importlib import metadata
from pathlib import Path

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from klask_common import CTRL_DT, parse_backend
from klask_env import KlaskSelfPlayEnv
from klask_eval_lib import baseline_actions


def resolve_checkpoint(log_dir, ckpt):
    exact = log_dir / f"model_{ckpt}.pt"
    if exact.exists():
        return exact
    candidates = []
    for path in log_dir.glob("model_*.pt"):
        try:
            step = int(path.stem.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        if ckpt < 0 or step <= ckpt:
            candidates.append((step, path))
    if candidates:
        step, path = max(candidates, key=lambda item: item[0])
        print(f"Checkpoint model_{ckpt}.pt not found; using model_{step}.pt.")
        return path
    raise FileNotFoundError(f"No checkpoint files found in {log_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="klask_selfplay")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument(
        "--best",
        action="store_true",
        default=False,
        help="load best_model.pt (selected during training) instead of --ckpt",
    )
    parser.add_argument("--num_boards", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--opponent", choices=("self", "heuristic", "random", "passive"), default="self")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "cpu", "gpu", "cuda", "metal", "amdgpu"),
        help="Genesis backend override",
    )
    parser.add_argument("--record_width", type=int, default=1280)
    parser.add_argument("--record_height", type=int, default=720)
    parser.add_argument("--video_dir", type=Path, default=None)
    args = parser.parse_args()

    gs.init(backend=parse_backend(args.backend), precision="32", seed=args.seed)

    log_dir = Path("logs") / args.exp_name
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, train_cfg = pickle.load(f)

    env_cfg["num_boards"] = args.num_boards
    env_cfg["rendered_envs"] = min(args.num_boards, 4 if args.vis else 1)
    if args.record:
        env_cfg["record_camera"] = True
        env_cfg["record_res"] = (args.record_width, args.record_height)

    env = KlaskSelfPlayEnv(
        num_boards=args.num_boards,
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.best:
        ckpt_path = log_dir / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} not found (run training with periodic eval first).")
    else:
        ckpt_path = resolve_checkpoint(log_dir, args.ckpt)
    print(f"Loading checkpoint {ckpt_path}")
    runner.load(ckpt_path, map_location=gs.device)
    policy = runner.get_inference_policy(device=gs.device)

    video_dir = args.video_dir or (log_dir / "eval_videos")
    if args.record:
        video_dir.mkdir(parents=True, exist_ok=True)
        env.record_cam.start_recording()

    obs_dict = env.reset()
    ctrl_dt = env_cfg.get("ctrl_dt", CTRL_DT)
    max_steps = args.steps or int(env_cfg["episode_length_s"] / ctrl_dt)
    left_wins = 0
    right_wins = 0
    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs_dict)
            if args.opponent != "self":
                right_obs = obs_dict["policy"][args.num_boards :]
                actions = actions.clone()
                actions[args.num_boards :] = baseline_actions(right_obs, args.opponent)

            obs_dict, _, dones, infos = env.step(actions)
            if args.record:
                env.record_cam.render()
            if args.vis:
                # Viewer stepping is driven by the scene; this keeps the loop readable on fast machines.
                pass

            done_boards = dones[: args.num_boards].bool().nonzero(as_tuple=False).reshape((-1,))
            if len(done_boards) > 0:
                scored_by = env.last_scored_by.detach().cpu().tolist()
                for board_idx in done_boards.detach().cpu().tolist():
                    reason = env.last_score_reason[board_idx]
                    if reason == "none":
                        continue
                    if scored_by[board_idx] == 0:
                        left_wins += 1
                    elif scored_by[board_idx] == 1:
                        right_wins += 1

    if args.record:
        output_path = video_dir / "klask_eval.mp4"
        env.record_cam.stop_recording(save_to_filename=str(output_path), fps=int(1.0 / ctrl_dt))
        print(f"Saved {output_path}")

    print(f"Eval finished. Approx score: left {left_wins} - right {right_wins}")


if __name__ == "__main__":
    main()
