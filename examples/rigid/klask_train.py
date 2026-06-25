import argparse
import csv
import json
import math
import pickle
import shutil
import traceback
from importlib import metadata
from pathlib import Path

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from klask_common import parse_backend
from klask_env import REWARD_PROFILES, KlaskSelfPlayEnv, get_default_env_cfg, get_reward_cfg
from klask_eval_lib import rollout_eval


def get_train_cfg(exp_name, num_steps_per_env, save_interval):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.985,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": num_steps_per_env,
        "save_interval": save_interval,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def find_latest_checkpoint(log_dir):
    """Return (path, iter) of the highest-numbered model_*.pt in log_dir, or (None, -1)."""
    best_path, best_step = None, -1
    for path in log_dir.glob("model_*.pt"):
        try:
            step = int(path.stem.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        if step > best_step:
            best_path, best_step = path, step
    return best_path, best_step


CSV_FIELDS = (
    "iter",
    "opponent",
    "games",
    "decisive",
    "draws",
    "policy_scores",
    "opp_scores",
    "win_rate",
    "net_score",
    "net_score_per_game",
    "policy_goals",
    "out_of_bounds",
    "biscuit_scores",
    "mean_left_reward",
    "is_best",
    "video_path",
)


def run_periodic_eval(env, runner, args, ctrl_dt, log_dir, video_dir, eval_writer, csv_path, state):
    """Evaluate the freshly-trained in-memory policy, log metrics, and track best.

    Returns the metric value, or None if the evaluation failed. Any failure here is
    swallowed so it can never abort the long training run.
    """
    it = runner.current_learning_iteration
    try:
        policy = runner.get_inference_policy(device=gs.device)
        record = args.eval_video and not args.vis
        video_path = (video_dir / f"eval_iter_{it:04d}.mp4") if record else None
        metrics = rollout_eval(
            env,
            policy,
            opponent=args.eval_opponent,
            num_steps=args.eval_steps,
            record=record,
            record_fps=round(1.0 / ctrl_dt),
            video_path=video_path,
        )
    except Exception:
        state["fail_count"] += 1
        print(f"[eval] iteration {it}: evaluation failed (failure #{state['fail_count']}):")
        traceback.print_exc()
        if state["fail_count"] >= 2 and args.eval_video:
            print("[eval] disabling video recording after repeated failures; metrics-only from now on.")
            args.eval_video = False
        return None

    state["fail_count"] = 0
    metric_val = metrics[args.eval_metric]
    # All bookkeeping below (best-model copy, json/csv writes, tensorboard) is guarded so a
    # transient filesystem error can never abort the long training run.
    try:
        latest_ckpt = log_dir / f"model_{it}.pt"
        # Only let an eval compete for "best" once it has produced enough decisive games.
        # Otherwise a degenerate early rollout (0 games -> win_rate defaults to 0.5) could
        # permanently lock best_model.pt to a useless early checkpoint.
        eligible = metrics["decisive"] >= args.min_eval_games
        is_best = eligible and metric_val > state["best_metric"] and latest_ckpt.exists()
        if is_best:
            state["best_metric"] = metric_val
            state["best_iter"] = it
            shutil.copy(latest_ckpt, log_dir / "best_model.pt")
            if metrics.get("video_path") and Path(metrics["video_path"]).exists():
                shutil.copy(metrics["video_path"], log_dir / "best_eval.mp4")
            with open(log_dir / "best_model_info.json", "w") as f:
                json.dump(
                    {"iter": it, "metric_name": args.eval_metric, "metric_value": metric_val, "metrics": metrics},
                    f,
                    indent=2,
                )

        if eval_writer is not None:
            for key in (
                "win_rate",
                "net_score",
                "net_score_per_game",
                "games",
                "policy_scores",
                "opp_scores",
                "policy_goals",
                "out_of_bounds",
                "mean_left_reward",
            ):
                eval_writer.add_scalar(f"Eval/{key}", float(metrics[key]), it)
            eval_writer.add_scalar("Eval/best_metric", float(state["best_metric"]), it)
            eval_writer.flush()

        row = {field: metrics.get(field, "") for field in CSV_FIELDS}
        row["iter"] = it
        row["is_best"] = int(is_best)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        star = " *BEST*" if is_best else (" (too few games)" if not eligible else "")
        print(
            f"[eval] iter {it} vs {args.eval_opponent}: "
            f"win_rate={metrics['win_rate']:.3f} net={metrics['net_score']} "
            f"(policy {metrics['policy_scores']} / opp {metrics['opp_scores']}, {metrics['games']} games) "
            f"{args.eval_metric}={metric_val:.4f}{star}"
        )
    except Exception:
        print(f"[eval] iteration {it}: post-rollout bookkeeping failed; continuing training:")
        traceback.print_exc()
    return metric_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="klask_selfplay")
    parser.add_argument("-B", "--num_boards", type=int, default=256)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--num_steps_per_env", type=int, default=128)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "cpu", "gpu", "cuda", "metal", "amdgpu"),
        help="Genesis backend override",
    )
    parser.add_argument("--keep_logs", action="store_true", help="do not delete an existing log directory")
    parser.add_argument(
        "--reward_profile",
        choices=tuple(REWARD_PROFILES),
        default="balanced",
        help="reward shaping profile ('balanced' = dense; 'simple' = sparse, anti-biscuit)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from the latest model_*.pt in the log dir (implies --keep_logs)",
    )
    # Periodic in-training evaluation / best-model selection / video recording.
    parser.add_argument("--no_eval", action="store_true", default=False, help="disable periodic evaluation")
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=0,
        help="iterations between evaluations (0 = use --save_interval)",
    )
    parser.add_argument("--eval_steps", type=int, default=300, help="control steps per evaluation rollout")
    parser.add_argument(
        "--eval_opponent",
        choices=("self", "heuristic", "random", "passive"),
        default="heuristic",
        help="opponent the policy is scored against when picking the best model",
    )
    parser.add_argument(
        "--eval_metric",
        choices=("win_rate", "net_score", "net_score_per_game"),
        default="win_rate",
        help="metric used to select best_model.pt",
    )
    parser.add_argument(
        "--min_eval_games",
        type=int,
        default=10,
        help="minimum decisive games in an eval before it can be selected as best",
    )
    parser.add_argument("--no_eval_video", dest="eval_video", action="store_false", default=True)
    parser.add_argument("--record_width", type=int, default=960)
    parser.add_argument("--record_height", type=int, default=540)
    args = parser.parse_args()

    gs.init(
        backend=parse_backend(args.backend),
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=not args.vis,
    )

    log_dir = Path("logs") / args.exp_name
    if log_dir.exists() and not args.keep_logs and not args.resume:
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfgs_path = log_dir / "cfgs.pkl"
    if args.resume and cfgs_path.exists():
        # Preserve the exact reward + physics the checkpoint was trained with; silently
        # switching the reward profile on resume would corrupt the run's objective.
        with open(cfgs_path, "rb") as f:
            env_cfg, reward_cfg, _ = pickle.load(f)
        reward_cfg.setdefault("own_side_penalty", 0.0)
        print(f"Resume: loaded saved reward+env cfg from {cfgs_path} (reward profile preserved).")
    else:
        env_cfg = get_default_env_cfg(args.num_boards)
        reward_cfg = get_reward_cfg(args.reward_profile)
        print(f"Reward profile: {args.reward_profile}")

    # Session-level settings that may legitimately change between (re)launches.
    env_cfg["num_boards"] = args.num_boards
    env_cfg.pop("record_camera", None)
    env_cfg.pop("record_res", None)
    if args.vis:
        env_cfg["rendered_envs"] = min(4, args.num_boards)

    eval_enabled = not args.no_eval
    record_video = eval_enabled and args.eval_video and not args.vis
    if record_video:
        # Render board 0 only for a clean single-board evaluation clip.
        env_cfg["record_camera"] = True
        env_cfg["record_res"] = (args.record_width, args.record_height)
        env_cfg["rendered_envs"] = 1

    save_interval = max(1, min(args.save_interval, args.max_iterations))
    train_cfg = get_train_cfg(args.exp_name, args.num_steps_per_env, save_interval)

    with open(cfgs_path, "wb") as f:
        pickle.dump([env_cfg, reward_cfg, train_cfg], f)

    env = KlaskSelfPlayEnv(
        num_boards=args.num_boards,
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Resume support: pick up from the latest checkpoint and restore best-model bookkeeping.
    start_iter = 0
    resumed_best_metric = -math.inf
    resumed_best_iter = -1
    if args.resume:
        ckpt_path, ckpt_step = find_latest_checkpoint(log_dir)
        if ckpt_path is not None:
            try:
                runner.load(str(ckpt_path), map_location=gs.device)
                start_iter = runner.current_learning_iteration + 1
                runner.current_learning_iteration = start_iter
                info_path = log_dir / "best_model_info.json"
                if info_path.exists():
                    with open(info_path) as f:
                        info = json.load(f)
                    resumed_best_metric = info.get("metric_value", -math.inf)
                    resumed_best_iter = info.get("iter", -1)
                print(f"Resuming from {ckpt_path.name} -> next iteration {start_iter} "
                      f"(best so far: {resumed_best_metric} @ iter {resumed_best_iter}).")
            except Exception:
                print(f"Failed to resume from {ckpt_path}; starting fresh:")
                traceback.print_exc()
                start_iter = 0
        else:
            print("--resume requested but no checkpoint found; starting fresh.")

    if start_iter >= args.max_iterations:
        print(f"Already trained {start_iter} >= {args.max_iterations} iterations; nothing to do.")
        return

    if not eval_enabled:
        runner.learn(num_learning_iterations=args.max_iterations - start_iter, init_at_random_ep_len=True)
        return

    ctrl_dt = env_cfg["ctrl_dt"]
    eval_interval = args.eval_interval if args.eval_interval > 0 else save_interval
    eval_interval = max(1, min(eval_interval, args.max_iterations))

    video_dir = log_dir / "eval_videos"
    if record_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter

    eval_writer = SummaryWriter(log_dir=str(log_dir))
    csv_path = log_dir / "eval_results.csv"
    state = {"best_metric": resumed_best_metric, "best_iter": resumed_best_iter, "fail_count": 0}

    print(
        f"Training iterations {start_iter}..{args.max_iterations - 1} in chunks of {eval_interval}; "
        f"evaluating vs '{args.eval_opponent}' (metric={args.eval_metric}, "
        f"video={'on' if record_video else 'off'})."
    )

    total_done = start_iter
    first_chunk = True
    while total_done < args.max_iterations:
        chunk = min(eval_interval, args.max_iterations - total_done)
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=first_chunk)
        first_chunk = False
        total_done += chunk

        # Evaluation is auxiliary; no failure in it may abort the long training run.
        try:
            run_periodic_eval(env, runner, args, ctrl_dt, log_dir, video_dir, eval_writer, csv_path, state)
        except Exception:
            print(f"[eval] unexpected error after iteration {runner.current_learning_iteration}; continuing:")
            traceback.print_exc()

        # Re-seed episode phases so boards stay desynchronised. Match rsl_rl's inference-mode
        # env driving so in-place buffer writes stay legal.
        try:
            with torch.inference_mode():
                env.reset()
                env.episode_length_buf = torch.randint_like(
                    env.episode_length_buf, high=int(env.max_episode_length)
                )
        except Exception:
            print("[eval] episode reseed failed; continuing:")
            traceback.print_exc()

        # Always advance the counter past the boundary so the next chunk neither
        # re-runs nor skips an iteration, regardless of any eval/reseed failure above.
        runner.current_learning_iteration = total_done

    eval_writer.flush()
    eval_writer.close()
    if state["best_iter"] >= 0:
        print(
            f"Best model: iter {state['best_iter']} with {args.eval_metric}={state['best_metric']:.4f} "
            f"-> {log_dir / 'best_model.pt'}"
        )
    else:
        print("No best model selected (evaluation never produced a valid metric).")


if __name__ == "__main__":
    main()
