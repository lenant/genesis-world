import argparse
import pickle
import re
import time
from importlib import metadata
from pathlib import Path

import torch

import genesis as gs

from behavior_cloning import Policy
from so100_grasp_env import GraspEnv

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner


def load_rl_policy(env, train_cfg, log_dir):
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    runner.load(last_ckpt, map_location=gs.device)
    print(f"Loaded RL checkpoint from {last_ckpt}")
    return runner.get_inference_policy(device=gs.device)


def load_bc_policy_for_inference(bc_cfg, log_dir, action_dim):
    """Load the BC policy without allocating the training replay buffer."""
    policy = Policy(bc_cfg["policy"], action_dim).to(gs.device)

    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
    checkpoint = torch.load(last_ckpt, map_location=gs.device, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    print(f"Loaded BC checkpoint from {last_ckpt}")
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="so100_grasp")
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "bc"],
        help="Model type: 'rl' for reinforcement learning, 'bc' for behavior cloning",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record stereo camera videos during evaluation",
    )
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        help="Show the interactive Genesis viewer during evaluation.",
    )
    parser.add_argument(
        "--record_policy_cameras",
        action="store_true",
        help="Also record the low-resolution policy input cameras.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments to run during evaluation",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=None,
        help="Directory for recorded videos. Defaults to the experiment log directory.",
    )
    parser.add_argument(
        "--record_width",
        type=int,
        default=640,
        help="Width for recorded camera videos.",
    )
    parser.add_argument(
        "--record_height",
        type=int,
        default=480,
        help="Height for recorded camera videos.",
    )
    parser.add_argument(
        "--record_renderer",
        type=str,
        choices=["rasterizer", "nyx"],
        default="rasterizer",
        help="Renderer used only for saved evaluation videos.",
    )
    parser.add_argument(
        "--nyx_spp",
        type=int,
        default=32,
        help="Nyx samples per pixel for recorded videos.",
    )
    parser.add_argument(
        "--nyx_env_map",
        type=Path,
        default=Path("../genesis-nyx/examples/assets/kloppenheim_07_puresky_4k.hdr"),
        help="HDRI environment map used by Nyx recording, if the file exists.",
    )
    parser.add_argument(
        "--nyx_env_multiplier",
        type=float,
        default=2.0,
        help="Brightness multiplier for the Nyx HDRI environment map.",
    )
    parser.add_argument(
        "--scene_style",
        type=str,
        choices=["plain", "studio"],
        default=None,
        help="Visual scene style for recorded videos. Defaults to studio for Nyx and plain otherwise.",
    )
    parser.add_argument(
        "--skip_demo",
        action="store_true",
        help="Skip the scripted grasp-and-lift segment after policy rollout",
    )
    args = parser.parse_args()

    gs.init()

    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    video_dir = args.video_dir or log_dir

    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)

    env_cfg["num_envs"] = args.num_envs
    env_cfg["box_fixed"] = False
    env_cfg["visualize_camera"] = False
    if args.record:
        env_cfg["record_image_resolution"] = (args.record_width, args.record_height)
        env_cfg["record_renderer"] = args.record_renderer
        env_cfg["record_scene_style"] = args.scene_style or ("studio" if args.record_renderer == "nyx" else "plain")

        if args.record_renderer == "nyx":
            env_cfg["record_cameras"] = {
                "record_front_cam": {
                    "pos": (0.45, -0.85, 0.45),
                    "lookat": (0.0, -0.28, 0.08),
                    "fov": 48,
                },
                "record_side_cam": {
                    "pos": (0.55, -0.22, 0.35),
                    "lookat": (0.0, -0.28, 0.08),
                    "fov": 52,
                },
            }
            env_cfg["nyx"] = {
                "spp": args.nyx_spp,
                "env_map_multiplier": args.nyx_env_multiplier,
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
            if args.nyx_env_map.exists():
                env_cfg["nyx"]["env_map"] = str(args.nyx_env_map)

    if args.record:
        video_dir.mkdir(parents=True, exist_ok=True)
        if args.record_renderer == "nyx":
            env_cfg["record_video"] = {
                "record_front_cam": str(video_dir / "nyx_front_cam.mp4"),
                "record_side_cam": str(video_dir / "nyx_side_cam.mp4"),
            }
        else:
            env_cfg["record_video"] = {
                "record_left_cam": str(video_dir / "headless_left_cam.mp4"),
                "record_right_cam": str(video_dir / "headless_right_cam.mp4"),
            }
        if args.record_policy_cameras:
            env_cfg["record_video"].update(
                {
                    "left_cam": str(video_dir / "policy_left_cam.mp4"),
                    "right_cam": str(video_dir / "policy_right_cam.mp4"),
                }
            )

    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy_for_inference(bc_train_cfg, log_dir, env.num_actions)

    obs_dict = env.reset()
    max_sim_step = int(env_cfg["episode_length_s"] / env_cfg["ctrl_dt"])

    rollout_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_sim_step):
            if args.stage == "rl":
                actions = policy(obs_dict)
            else:
                rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()
                actions = policy(rgb_obs, ee_pose)

            obs_dict, _, _, _ = env.step(actions)

        rollout_elapsed = time.perf_counter() - rollout_start
        print(
            f"Rollout throughput: {max_sim_step} steps in {rollout_elapsed:.3f}s "
            f"({max_sim_step / rollout_elapsed:.2f} steps/s)"
        )

        if not args.skip_demo:
            env.grasp_and_lift_demo()

        if args.record:
            env.scene.stop_recording()
            print(f"Saved videos to {video_dir}")


if __name__ == "__main__":
    main()
