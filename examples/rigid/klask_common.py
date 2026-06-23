import numpy as np

import genesis as gs


AGENTS = ("left", "right")
OPPONENT = {"left": "right", "right": "left"}

# Public KLASK references list a 30 x 40 cm playfield and an approximately 3 cm goal.
# The ball/striker/biscuit sizes and material parameters below are real-scale approximations.
SIM_DT = 1 / 240
FRAME_SKIP = 4
CTRL_DT = SIM_DT * FRAME_SKIP

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
BISCUIT_MAX_SPEED = 1.2

LEFT_START = np.array([-0.11, 0.0], dtype=np.float32)
RIGHT_START = np.array([0.11, 0.0], dtype=np.float32)
BALL_START_MIN_X = 0.035
BALL_START_MAX_X = 0.075
BALL_START_Y_RANGE = 0.02

BOARD_COLOR = (0.07, 0.34, 0.36, 1.0)
RAIL_COLOR = (0.93, 0.91, 0.83, 1.0)
LINE_COLOR = (0.82, 0.88, 0.83, 1.0)
LEFT_COLOR = (0.92, 0.32, 0.29, 1.0)
RIGHT_COLOR = (0.29, 0.49, 0.92, 1.0)
BALL_COLOR = (0.96, 0.96, 0.93, 1.0)
BISCUIT_COLOR = (0.42, 0.43, 0.43, 1.0)
GOAL_COLOR = (0.16, 0.16, 0.16, 1.0)

MAGNET_FEATURES = 5
OBSERVATION_SIZE = 16 + len(BISCUIT_Y) * MAGNET_FEATURES + 2


def parse_backend(name):
    if name == "auto":
        return None
    return getattr(gs, name)


def side_sign(side):
    return 1.0 if side == "left" else -1.0


def canonical_action_to_world(side, action):
    action = np.asarray(action, dtype=np.float32)
    if side == "left":
        return action
    return np.array([-action[0], action[1]], dtype=np.float32)


def striker_x_bounds(side):
    half_w = BOARD_WIDTH / 2.0
    if side == "left":
        return (-half_w + STRIKER_RADIUS, -STRIKER_RADIUS)
    return (STRIKER_RADIUS, half_w - STRIKER_RADIUS)


def striker_y_bounds():
    half_h = BOARD_HEIGHT / 2.0
    return (-half_h + STRIKER_RADIUS, half_h - STRIKER_RADIUS)


def biscuit_bounds():
    half_w = BOARD_WIDTH / 2.0
    half_h = BOARD_HEIGHT / 2.0
    return (
        -half_w + BISCUIT_RADIUS,
        half_w - BISCUIT_RADIUS,
        -half_h + BISCUIT_RADIUS,
        half_h - BISCUIT_RADIUS,
    )


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


def add_striker(scene, side):
    start = LEFT_START if side == "left" else RIGHT_START
    color = LEFT_COLOR if side == "left" else RIGHT_COLOR
    return add_cylinder(
        scene,
        (start[0], start[1], STRIKER_Z),
        STRIKER_RADIUS,
        STRIKER_HEIGHT,
        color,
        rho=5000,
        gravity_compensation=1.0,
    )


def add_ball(scene, xy=(0.07, 0.02)):
    return add_sphere(scene, (xy[0], xy[1], BALL_Z), BALL_RADIUS, BALL_COLOR, rho=1000, friction=0.16)


def add_biscuits(scene):
    return [
        add_cylinder(scene, (0.0, y, BISCUIT_Z), BISCUIT_RADIUS, BISCUIT_HEIGHT, BISCUIT_COLOR, rho=800, friction=0.4)
        for y in BISCUIT_Y
    ]


def create_scene(
    *,
    show_viewer=False,
    rendered_envs=1,
    max_collision_pairs=256,
    show_fps=False,
    sim_dt=SIM_DT,
    sim_substeps=4,
    constraint_timeconst=0.004,
):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            box_box_detection=True,
            constraint_timeconst=constraint_timeconst,
            max_collision_pairs=max_collision_pairs,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -0.46, 0.32),
            camera_lookat=(0.0, 0.0, 0.02),
            camera_fov=50,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=list(range(max(1, rendered_envs))),
            show_world_frame=False,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=show_fps),
        show_viewer=show_viewer,
    )


def set_body_state(entity, xy, z, *, zero_velocity=True):
    entity.set_pos((float(xy[0]), float(xy[1]), z), zero_velocity=zero_velocity)
    entity.set_quat((1.0, 0.0, 0.0, 0.0), zero_velocity=False)
    entity.set_dofs_velocity(np.zeros(entity.n_dofs, dtype=np.float32))
