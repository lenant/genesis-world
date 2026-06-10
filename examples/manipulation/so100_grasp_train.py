import argparse
import pickle
import re
from importlib import metadata
from pathlib import Path

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner
from behavior_cloning import BehaviorCloning

import genesis as gs

from so100_grasp_env import GraspEnv


def get_train_cfg(exp_name):
    # stage 1: privileged reinforcement learning
    rl_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
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
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "relu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    # stage 2: vision-based behavior cloning
    bc_cfg_dict = {
        # Basic training parameters
        "num_steps_per_env": 24,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "num_mini_batches": 10,
        "max_grad_norm": 1.0,
        # Network architecture
        "policy": {
            "vision_encoder": {
                "conv_layers": [
                    {
                        "in_channels": 3,  # 3 channel for rgb image
                        "out_channels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                    {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ],
                "pooling": "adaptive_avg",
            },
            "action_head": {
                "state_obs_dim": 7,  # end-effector pose as additional state observation
                "hidden_dims": [128, 128, 64],
            },
            "pose_head": {
                "hidden_dims": [64, 64],
            },
        },
        # Training settings
        "buffer_size": 1000,
        "log_freq": 10,
        "save_freq": 50,
        "eval_freq": 50,
    }

    return rl_cfg_dict, bc_cfg_dict


def get_task_cfgs():
    env_cfg = {
        "num_envs": 10,
        "num_actions": 5,
        "action_scales": [0.035, 0.035, 0.035, 0.035, 0.035],
        "episode_length_s": 3.0,
        "ctrl_dt": 0.01,
        "box_size": [0.04, 0.04, 0.04],
        "object_x_bounds": (-0.12, 0.12),
        "object_y_bounds": (-0.32, -0.24),
        "object_z": 0.02,
        "keypoint_unit_length": 0.04,
        "scripted_lift_height": 0.10,
        "image_resolution": (64, 64),
        "policy_cameras": {
            "left_cam": {
                "pos": (0.35, -0.75, 0.35),
                "lookat": (0.0, -0.28, 0.08),
                "fov": 55,
            },
            "right_cam": {
                "pos": (0.45, -0.22, 0.32),
                "lookat": (0.0, -0.28, 0.08),
                "fov": 55,
            },
        },
        "visualize_camera": False,
    }
    reward_scales = {
        "keypoints": 1.0,
    }
    # SO-100 robot specific
    robot_cfg = {
        "mjcf_file": "xml/so_arm100/so_arm100.xml",
        "arm_joint_names": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        "gripper_joint_names": ["Jaw"],
        "ee_link_name": "Fixed_Jaw",
        "gripper_link_names": ["Fixed_Jaw", "Moving_Jaw"],
        "default_arm_dof": [0.0, -1.57079, 1.57079, 1.57079, -1.57079],
        "default_gripper_dof": [0.65],
        "gripper_open_dof": [0.65],
        "gripper_close_dof": [0.0],
        "arm_lower_limits": [-2.2, -3.14158, 0.0, -2.0, -3.14158],
        "arm_upper_limits": [2.2, 0.2, 3.14158, 1.8, 3.14158],
        "fixed_finger_tip_offset": [0.012, -0.08, 0.0],
        "moving_finger_tip_offset": [-0.009, -0.055, 0.0],
        "dof_kp": [80, 80, 60, 40, 30, 20],
        "dof_kv": [8, 8, 6, 4, 3, 2],
        "dof_force_lower": [-35, -35, -35, -35, -35, -15],
        "dof_force_upper": [35, 35, 35, 35, 35, 15],
    }
    return env_cfg, reward_scales, robot_cfg


def load_teacher_policy(env, rl_train_cfg, exp_name):
    # load teacher policy
    log_dir = Path("logs") / f"{exp_name + '_' + 'rl'}"
    assert log_dir.exists(), f"Log directory {log_dir} does not exist"
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
    runner.load(last_ckpt, map_location=gs.device)
    print(f"Loaded teacher policy from checkpoint {last_ckpt} from {log_dir}")
    teacher_policy = runner.get_inference_policy(device=gs.device)
    return teacher_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="so100_grasp")
    parser.add_argument(
        "--teacher_exp_name",
        type=str,
        default=None,
        help="Experiment name to load the RL teacher from for BC. Defaults to --exp_name.",
    )
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--stage", type=str, default="rl")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--bc_renderer",
        type=str,
        choices=["auto", "rasterizer", "nyx"],
        default="auto",
        help="Renderer used for BC policy image observations.",
    )
    parser.add_argument(
        "--bc_num_envs",
        type=int,
        default=None,
        help="Number of BC collection envs. Defaults to 1 for Nyx and 10 otherwise.",
    )
    parser.add_argument(
        "--nyx_spp",
        type=int,
        default=1,
        help="Nyx samples per pixel for BC policy cameras.",
    )
    parser.add_argument(
        "--scene_style",
        type=str,
        choices=["plain", "studio"],
        default=None,
        help="Visual scene style. Defaults to studio for Nyx BC and plain otherwise.",
    )
    args = parser.parse_args()

    # === init ===
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    # === task cfgs and trainning algos cfgs ===
    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    rl_train_cfg, bc_train_cfg = get_train_cfg(args.exp_name)

    # === log dir ===
    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # === env ===
    # BC only needs a small number of envs, e.g., 10
    if args.stage == "rl":
        env_cfg["num_envs"] = args.num_envs
    else:
        default_bc_num_envs = 1 if args.bc_renderer == "nyx" else 10
        env_cfg["num_envs"] = args.bc_num_envs or default_bc_num_envs
        env_cfg["policy_renderer"] = args.bc_renderer
        env_cfg["record_scene_style"] = args.scene_style or ("studio" if args.bc_renderer == "nyx" else "plain")
        if args.bc_renderer == "nyx":
            bc_train_cfg["log_freq"] = 1
            env_cfg["nyx"] = {
                "spp": args.nyx_spp,
                "env_map_multiplier": 2.0,
                "lights": [
                    {
                        "type": "directional",
                        "dir": (-0.35, -0.25, -0.9),
                        "color": (1.0, 0.96, 0.9),
                        "intensity": 4.5,
                        "shadow": True,
                    },
                    {
                        "type": "directional",
                        "dir": (0.55, 0.35, -0.75),
                        "color": (0.7, 0.78, 1.0),
                        "intensity": 0.9,
                        "shadow": False,
                    },
                ],
            }

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg), f)
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # === runner ===
    if args.stage == "bc":
        teacher_policy = load_teacher_policy(env, rl_train_cfg, args.teacher_exp_name or args.exp_name)
        runner = BehaviorCloning(env, bc_train_cfg, teacher_policy, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=log_dir)
    else:
        runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training

# to train the SO-100 RL policy
python examples/manipulation/so100_grasp_train.py --stage=rl

# to train the SO-100 BC policy (requires RL policy to be trained first)
python examples/manipulation/so100_grasp_train.py --stage=bc
"""
