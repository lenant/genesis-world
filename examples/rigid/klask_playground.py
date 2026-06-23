import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


# Public KLASK references list a 30 x 40 cm playfield and an approximately 3 cm goal.
# The ball/striker/biscuit sizes and material parameters below are real-scale approximations.
SIM_DT = 1 / 240
BOARD_WIDTH = 0.40
BOARD_HEIGHT = 0.30
GOAL_WIDTH = 0.03

STRIKER_RADIUS = 0.014
BALL_RADIUS = 0.005
BISCUIT_RADIUS = 0.0065
BISCUIT_Y = (-0.06, 0.0, 0.06)

TABLE_THICKNESS = 0.012
WALL_THICKNESS = 0.012
WALL_HEIGHT = 0.025
STRIKER_HEIGHT = 0.028
BISCUIT_HEIGHT = 0.008
GOAL_TRAY_DEPTH = 0.035
DISC_CLEARANCE = 0.0005

TABLE_CENTER_Z = TABLE_THICKNESS / 2.0
TABLE_TOP_Z = TABLE_THICKNESS
STRIKER_Z = TABLE_TOP_Z + STRIKER_HEIGHT / 2.0 + DISC_CLEARANCE
BALL_Z = TABLE_TOP_Z + BALL_RADIUS + DISC_CLEARANCE
BISCUIT_Z = TABLE_TOP_Z + BISCUIT_HEIGHT / 2.0 + DISC_CLEARANCE

HANDLE_SPEED = 4.95
MAX_HANDLE_VEL = 10.8
BALL_MAX_SPEED = 1.5

LEFT_START = np.array([-0.11, 0.0], dtype=np.float32)
RIGHT_START = np.array([0.11, 0.0], dtype=np.float32)

BOARD_COLOR = (0.07, 0.34, 0.36, 1.0)
RAIL_COLOR = (0.93, 0.91, 0.83, 1.0)
LINE_COLOR = (0.82, 0.88, 0.83, 1.0)
LEFT_COLOR = (0.92, 0.32, 0.29, 1.0)
RIGHT_COLOR = (0.29, 0.49, 0.92, 1.0)
BALL_COLOR = (0.96, 0.96, 0.93, 1.0)
BISCUIT_COLOR = (0.42, 0.43, 0.43, 1.0)
GOAL_COLOR = (0.16, 0.16, 0.16, 1.0)


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


def parse_backend(name):
    if name == "auto":
        return None
    return getattr(gs, name)


def add_box(scene, pos, size, color, *, collision=True, friction=0.4):
    return scene.add_entity(
        morph=gs.morphs.Box(pos=pos, size=size, fixed=True, collision=collision),
        material=gs.materials.Rigid(friction=friction),
        surface=gs.surfaces.Default(color=color),
    )


def add_cylinder(scene, pos, radius, height, color, *, rho=600, friction=0.35, gravity_compensation=0.0):
    return scene.add_entity(
        morph=gs.morphs.Cylinder(pos=pos, radius=radius, height=height),
        material=gs.materials.Rigid(rho=rho, friction=friction, gravity_compensation=gravity_compensation),
        surface=gs.surfaces.Default(color=color),
    )


def add_sphere(scene, pos, radius, color, *, rho=1000, friction=0.18):
    return scene.add_entity(
        morph=gs.morphs.Sphere(pos=pos, radius=radius),
        material=gs.materials.Rigid(rho=rho, friction=friction),
        surface=gs.surfaces.Default(color=color),
    )


def build_board(scene):
    half_w = BOARD_WIDTH / 2.0
    half_h = BOARD_HEIGHT / 2.0
    goal_half = GOAL_WIDTH / 2.0

    add_box(
        scene,
        (0.0, 0.0, TABLE_CENTER_Z),
        (BOARD_WIDTH, BOARD_HEIGHT, TABLE_THICKNESS),
        BOARD_COLOR,
        friction=0.18,
    )

    rail_z = TABLE_TOP_Z + WALL_HEIGHT / 2.0
    add_box(
        scene,
        (0.0, half_h + WALL_THICKNESS / 2.0, rail_z),
        (BOARD_WIDTH, WALL_THICKNESS, WALL_HEIGHT),
        RAIL_COLOR,
        friction=0.45,
    )
    add_box(
        scene,
        (0.0, -half_h - WALL_THICKNESS / 2.0, rail_z),
        (BOARD_WIDTH, WALL_THICKNESS, WALL_HEIGHT),
        RAIL_COLOR,
        friction=0.45,
    )

    side_segment = (BOARD_HEIGHT - GOAL_WIDTH) / 2.0
    for x in (-half_w - WALL_THICKNESS / 2.0, half_w + WALL_THICKNESS / 2.0):
        add_box(
            scene,
            (x, -(goal_half + side_segment / 2.0), rail_z),
            (WALL_THICKNESS, side_segment, WALL_HEIGHT),
            RAIL_COLOR,
            friction=0.45,
        )
        add_box(
            scene,
            (x, goal_half + side_segment / 2.0, rail_z),
            (WALL_THICKNESS, side_segment, WALL_HEIGHT),
            RAIL_COLOR,
            friction=0.45,
        )

    for sign in (-1.0, 1.0):
        tray_x = sign * (half_w + GOAL_TRAY_DEPTH / 2.0)
        add_box(
            scene,
            (tray_x, 0.0, TABLE_CENTER_Z),
            (GOAL_TRAY_DEPTH, GOAL_WIDTH, TABLE_THICKNESS),
            GOAL_COLOR,
            friction=0.35,
        )
        add_box(
            scene,
            (sign * (half_w + GOAL_TRAY_DEPTH), 0.0, rail_z),
            (WALL_THICKNESS, GOAL_WIDTH, WALL_HEIGHT),
            RAIL_COLOR,
            friction=0.45,
        )

    # Visual field markings only.
    add_box(
        scene,
        (0.0, 0.0, TABLE_TOP_Z + 0.001),
        (0.003, BOARD_HEIGHT * 0.88, 0.001),
        LINE_COLOR,
        collision=False,
    )
    add_box(
        scene,
        (-half_w + 0.03, 0.0, TABLE_TOP_Z + 0.001),
        (0.002, GOAL_WIDTH, 0.001),
        LINE_COLOR,
        collision=False,
    )
    add_box(
        scene,
        (half_w - 0.03, 0.0, TABLE_TOP_Z + 0.001),
        (0.002, GOAL_WIDTH, 0.001),
        LINE_COLOR,
        collision=False,
    )


def set_body_state(entity, xy, z, *, zero_velocity=True):
    entity.set_pos((float(xy[0]), float(xy[1]), z), zero_velocity=zero_velocity)
    entity.set_quat((1.0, 0.0, 0.0, 0.0), zero_velocity=False)
    entity.set_dofs_velocity(np.zeros(entity.n_dofs, dtype=np.float32))


def reset_round(left, right, ball, biscuits, left_controller, right_controller, rng):
    left_controller.reset()
    right_controller.reset()
    set_body_state(left, LEFT_START, STRIKER_Z)
    set_body_state(right, RIGHT_START, STRIKER_Z)

    ball_x = rng.choice((-1.0, 1.0)) * rng.uniform(0.05, 0.10)
    ball_y = rng.uniform(-0.02, 0.02)
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
    if abs(pos[0] - clamped_xy[0]) > 1e-4 or abs(pos[1] - clamped_xy[1]) > 1e-4 or abs(pos[2] - STRIKER_Z) > 0.003:
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

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=4, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            box_box_detection=True,
            constraint_timeconst=0.004,
            max_collision_pairs=256,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -0.46, 0.32),
            camera_lookat=(0.0, 0.0, 0.02),
            camera_fov=50,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )

    build_board(scene)
    left = add_cylinder(
        scene,
        (LEFT_START[0], LEFT_START[1], STRIKER_Z),
        STRIKER_RADIUS,
        STRIKER_HEIGHT,
        LEFT_COLOR,
        rho=5000,
        gravity_compensation=1.0,
    )
    right = add_cylinder(
        scene,
        (RIGHT_START[0], RIGHT_START[1], STRIKER_Z),
        STRIKER_RADIUS,
        STRIKER_HEIGHT,
        RIGHT_COLOR,
        rho=5000,
        gravity_compensation=1.0,
    )
    ball = add_sphere(scene, (0.07, 0.02, BALL_Z), BALL_RADIUS, BALL_COLOR, rho=1000, friction=0.16)
    biscuits = [
        add_cylinder(scene, (0.0, y, BISCUIT_Z), BISCUIT_RADIUS, BISCUIT_HEIGHT, BISCUIT_COLOR, rho=800, friction=0.4)
        for y in BISCUIT_Y
    ]

    scene.build()

    half_w = BOARD_WIDTH / 2.0
    half_h = BOARD_HEIGHT / 2.0
    goal_half = GOAL_WIDTH / 2.0
    handle_y_bounds = (-half_h + STRIKER_RADIUS, half_h - STRIKER_RADIUS)
    left_controller = PlayerController(
        LEFT_START,
        (-half_w + STRIKER_RADIUS, -STRIKER_RADIUS),
        handle_y_bounds,
    )
    right_controller = PlayerController(
        RIGHT_START,
        (STRIKER_RADIUS, half_w - STRIKER_RADIUS),
        handle_y_bounds,
    )
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
