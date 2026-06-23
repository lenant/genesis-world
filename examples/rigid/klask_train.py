import argparse
import pickle
import shutil
from importlib import metadata
from pathlib import Path

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from klask_common import parse_backend
from klask_env import KlaskSelfPlayEnv, get_default_env_cfg, get_default_reward_cfg


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
    args = parser.parse_args()

    gs.init(
        backend=parse_backend(args.backend),
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=not args.vis,
    )

    log_dir = Path("logs") / args.exp_name
    if log_dir.exists() and not args.keep_logs:
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = get_default_env_cfg(args.num_boards)
    reward_cfg = get_default_reward_cfg()
    if args.vis:
        env_cfg["rendered_envs"] = min(4, args.num_boards)
    save_interval = max(1, min(args.save_interval, args.max_iterations))
    train_cfg = get_train_cfg(args.exp_name, args.num_steps_per_env, save_interval)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, reward_cfg, train_cfg], f)

    env = KlaskSelfPlayEnv(
        num_boards=args.num_boards,
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
