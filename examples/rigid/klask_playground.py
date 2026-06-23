import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

from klask_common import (
    BALL_MAX_SPEED,
    BALL_RADIUS,
    BALL_START_MAX_X,
    BALL_START_MIN_X,
    BALL_START_Y_RANGE,
    BALL_Z,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    BISCUIT_Y,
    BISCUIT_Z,
    GOAL_WIDTH,
    HANDLE_SPEED,
    LEFT_START,
    MAX_HANDLE_VEL,
    RIGHT_START,
    STRIKER_Z,
    add_ball,
    add_biscuits,
    add_striker,
    build_board,
    create_scene,
    parse_backend,
    set_body_state,
    striker_x_bounds,
    striker_y_bounds,
)


class PlayerController:
    def __init__(self, start_xy, x_bounds, y_bounds):
        self.start_xy = np.array(start_xy, dtype=np.float32)
        self.input_xy = np.zeros(2, dtype=np.float32)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def set_input(self, delta_xy, active):
        delta = np.array(delta_xy, dtype=np.float32)
        self.input_xy += delta if active else -delta
        self.input_xy = np.clip(self.input_xy, -1.0, 1.0)

    def reset(self):
        self.input_xy[:] = 0.0

    def velocity(self):
        direction = self.input_xy.copy()
        norm = np.linalg.norm(direction)
        if norm > 1.0:
            direction /= norm
        return direction * HANDLE_SPEED


def reset_round(left, right, ball, biscuits, left_controller, right_controller, rng):
    left_controller.reset()
    right_controller.reset()
    set_body_state(left, LEFT_START, STRIKER_Z)
    set_body_state(right, RIGHT_START, STRIKER_Z)

    ball_x = rng.choice((-1.0, 1.0)) * rng.uniform(BALL_START_MIN_X, BALL_START_MAX_X)
    ball_y = rng.uniform(-BALL_START_Y_RANGE, BALL_START_Y_RANGE)
    set_body_state(ball, (ball_x, ball_y), BALL_Z)

    for biscuit, y in zip(biscuits, BISCUIT_Y, strict=True):
        set_body_state(biscuit, (0.0, y), BISCUIT_Z)


def drive_handle(entity, controller):
    pos = entity.get_pos().detach().cpu().numpy()
    vel_xy = np.clip(controller.velocity(), -MAX_HANDLE_VEL, MAX_HANDLE_VEL)

    if pos[0] <= controller.x_bounds[0] and vel_xy[0] < 0.0:
        vel_xy[0] = 0.0
    elif pos[0] >= controller.x_bounds[1] and vel_xy[0] > 0.0:
        vel_xy[0] = 0.0

    if pos[1] <= controller.y_bounds[0] and vel_xy[1] < 0.0:
        vel_xy[1] = 0.0
    elif pos[1] >= controller.y_bounds[1] and vel_xy[1] > 0.0:
        vel_xy[1] = 0.0

    entity.set_dofs_velocity((float(vel_xy[0]), float(vel_xy[1]), 0.0, 0.0, 0.0, 0.0))

    clamped_xy = np.array(
        [
            np.clip(pos[0], *controller.x_bounds),
            np.clip(pos[1], *controller.y_bounds),
        ],
        dtype=np.float32,
    )
    z_off = abs(pos[2] - STRIKER_Z)
    if abs(pos[0] - clamped_xy[0]) > 1e-4 or abs(pos[1] - clamped_xy[1]) > 1e-4 or z_off > 0.003:
        entity.set_pos((float(clamped_xy[0]), float(clamped_xy[1]), STRIKER_Z), zero_velocity=False)
    entity.set_quat((1.0, 0.0, 0.0, 0.0), zero_velocity=False)


def clamp_ball_speed(ball):
    vel = ball.get_dofs_velocity().detach().cpu().numpy()
    speed = np.linalg.norm(vel[:2])
    if speed > BALL_MAX_SPEED:
        vel[:2] *= BALL_MAX_SPEED / speed
        ball.set_dofs_velocity(vel)


def register_controls(scene, left_controller, right_controller, reset_callback, stop_callback):
    def bind_axis(name, key, controller, delta):
        return (
            Keybind(f"{name}_press", key, KeyAction.PRESS, callback=controller.set_input, args=(delta, True)),
            Keybind(f"{name}_release", key, KeyAction.RELEASE, callback=controller.set_input, args=(delta, False)),
        )

    for keybind_name in ("world_frame", "save_image", "camera_rotation", "wireframe", "record_video"):
        try:
            scene.viewer.remove_keybind(keybind_name)
        except ValueError:
            pass

    scene.viewer.register_keybinds(
        *bind_axis("left_up", Key.W, left_controller, (0.0, 1.0)),
        *bind_axis("left_down", Key.S, left_controller, (0.0, -1.0)),
        *bind_axis("left_left", Key.A, left_controller, (-1.0, 0.0)),
        *bind_axis("left_right", Key.D, left_controller, (1.0, 0.0)),
        *bind_axis("right_up", Key.UP, right_controller, (0.0, 1.0)),
        *bind_axis("right_down", Key.DOWN, right_controller, (0.0, -1.0)),
        *bind_axis("right_left", Key.LEFT, right_controller, (-1.0, 0.0)),
        *bind_axis("right_right", Key.RIGHT, right_controller, (1.0, 0.0)),
        Keybind("reset_round", Key.R, KeyAction.RELEASE, callback=reset_callback),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop_callback),
        overwrite=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Playable 3D KLASK playground built from Genesis primitives.")
    parser.add_argument("-v", "--vis", action="store_true", help="open the interactive Genesis viewer")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="number of simulation steps; defaults to endless with --vis",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "cpu", "gpu", "cuda", "metal", "amdgpu"),
        help="Genesis backend override",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    max_steps = args.steps
    if max_steps is None and not args.vis:
        max_steps = 3000

    gs.init(backend=parse_backend(args.backend), precision="32")

    scene = create_scene(show_viewer=args.vis, rendered_envs=1)
    build_board(scene)
    left = add_striker(scene, "left")
    right = add_striker(scene, "right")
    ball = add_ball(scene)
    biscuits = add_biscuits(scene)

    scene.build()

    half_w = BOARD_WIDTH / 2.0
    goal_half = GOAL_WIDTH / 2.0
    left_controller = PlayerController(LEFT_START, striker_x_bounds("left"), striker_y_bounds())
    right_controller = PlayerController(RIGHT_START, striker_x_bounds("right"), striker_y_bounds())
    rng = np.random.default_rng(args.seed)
    scores = {"left": 0, "right": 0}

    def reset_callback():
        reset_round(left, right, ball, biscuits, left_controller, right_controller, rng)
        print("Round reset.")

    is_running = True

    def stop_callback():
        nonlocal is_running
        is_running = False

    reset_round(left, right, ball, biscuits, left_controller, right_controller, rng)

    if args.vis:
        register_controls(scene, left_controller, right_controller, reset_callback, stop_callback)
        print("KLASK controls: left W/A/S/D, right arrow keys, R reset, Esc quit.")

    step = 0
    try:
        while is_running and (max_steps is None or step < max_steps):
            drive_handle(left, left_controller)
            drive_handle(right, right_controller)
            clamp_ball_speed(ball)

            scene.step()
            step += 1

            ball_pos = ball.get_pos().detach().cpu().numpy()
            if ball_pos[0] < -half_w - BALL_RADIUS and abs(ball_pos[1]) <= goal_half:
                scores["right"] += 1
                print(f"Right scores. Score: left {scores['left']} - right {scores['right']}")
                reset_round(left, right, ball, biscuits, left_controller, right_controller, rng)
            elif ball_pos[0] > half_w + BALL_RADIUS and abs(ball_pos[1]) <= goal_half:
                scores["left"] += 1
                print(f"Left scores. Score: left {scores['left']} - right {scores['right']}")
                reset_round(left, right, ball, biscuits, left_controller, right_controller, rng)

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
