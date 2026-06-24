import math
from copy import deepcopy

import torch
from tensordict import TensorDict

import genesis as gs
from klask_common import (
    AGENTS,
    BALL_MAX_SPEED,
    BALL_RADIUS,
    BALL_START_MAX_X,
    BALL_START_MIN_X,
    BALL_START_Y_RANGE,
    BALL_Z,
    BISCUIT_MAX_SPEED,
    BISCUIT_RADIUS,
    BISCUIT_Y,
    BISCUIT_Z,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GOAL_WIDTH,
    HANDLE_SPEED,
    LEFT_START,
    MAX_HANDLE_VEL,
    OBSERVATION_SIZE,
    RIGHT_START,
    STRIKER_RADIUS,
    STRIKER_Z,
    add_ball,
    add_biscuits,
    add_striker,
    biscuit_bounds,
    build_board,
    create_scene,
    striker_x_bounds,
    striker_y_bounds,
)


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class KlaskSelfPlayEnv:
    def __init__(self, num_boards, env_cfg, reward_cfg, show_viewer=False):
        self.num_boards = num_boards
        self.num_envs = num_boards * 2
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.env_cfg = env_cfg
        self.reward_cfg = deepcopy(reward_cfg)
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.frame_skip = env_cfg["frame_skip"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)
        self.handle_speed = env_cfg["handle_speed"]
        self.ball_max_speed = env_cfg["ball_max_speed"]
        self.biscuit_max_speed = env_cfg["biscuit_max_speed"]
        self.clip_actions = env_cfg["clip_actions"]
        self.attach_frames = env_cfg["biscuit_attach_frames"]
        self.attach_distance = STRIKER_RADIUS + BISCUIT_RADIUS + env_cfg["biscuit_attach_margin"]
        self.biscuit_score_threshold = env_cfg["biscuit_score_threshold"]
        self.biscuit_attraction_range = env_cfg["biscuit_attraction_range"]

        rendered_envs = min(env_cfg.get("rendered_envs", 1), num_boards)
        self.scene = create_scene(
            show_viewer=show_viewer,
            rendered_envs=rendered_envs,
            max_collision_pairs=env_cfg["max_collision_pairs"],
            show_fps=env_cfg.get("show_fps", False),
            sim_dt=env_cfg["sim_dt"],
            sim_substeps=env_cfg["sim_substeps"],
            constraint_timeconst=env_cfg["constraint_timeconst"],
        )
        build_board(self.scene)
        self.left = add_striker(self.scene, "left")
        self.right = add_striker(self.scene, "right")
        self.ball = add_ball(self.scene)
        self.biscuits = add_biscuits(self.scene)
        self.record_cam = None
        if env_cfg.get("record_camera", False):
            self.record_cam = self.scene.add_camera(
                res=env_cfg.get("record_res", (1280, 720)),
                pos=env_cfg.get("record_pos", (0.0, -0.46, 0.32)),
                lookat=env_cfg.get("record_lookat", (0.0, 0.0, 0.02)),
                fov=env_cfg.get("record_fov", 50),
                GUI=False,
            )

        self.scene.build(n_envs=num_boards, env_spacing=(0.55, 0.45))

        self.left_start = torch.tensor(LEFT_START, dtype=gs.tc_float, device=self.device)
        self.right_start = torch.tensor(RIGHT_START, dtype=gs.tc_float, device=self.device)
        self.biscuit_start_y = torch.tensor(BISCUIT_Y, dtype=gs.tc_float, device=self.device)
        self.identity_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=gs.tc_float, device=self.device)

        self.left_action = torch.zeros((self.num_boards, 2), dtype=gs.tc_float, device=self.device)
        self.right_action = torch.zeros_like(self.left_action)
        self.left_contact = torch.zeros((self.num_boards,), dtype=torch.bool, device=self.device)
        self.right_contact = torch.zeros_like(self.left_contact)
        self.last_ball_toucher = torch.full((self.num_boards,), -1, dtype=torch.long, device=self.device)
        self.new_biscuit_attachments = torch.zeros((self.num_boards, 2), dtype=gs.tc_float, device=self.device)

        self.biscuit_attached_to = torch.full(
            (self.num_boards, len(BISCUIT_Y)),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.biscuit_contact_frames = torch.zeros(
            (self.num_boards, len(BISCUIT_Y), 2),
            dtype=torch.long,
            device=self.device,
        )
        self.biscuit_offsets = torch.zeros(
            (self.num_boards, len(BISCUIT_Y), 2),
            dtype=gs.tc_float,
            device=self.device,
        )

        self.rew_buf = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_int, device=self.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=self.device)
        self.obs_buf = torch.zeros((self.num_envs, OBSERVATION_SIZE), dtype=gs.tc_float, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=self.device)
        self.actions = torch.zeros_like(self.last_actions)

        self.reward_names = (
            "progress",
            "position",
            "speed",
            "contact",
            "distance",
            "defense",
            "danger",
            "time",
            "action",
            "biscuit_attach",
            "biscuit_pull",
            "terminal",
        )
        self.episode_sums = {
            name: torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device) for name in self.reward_names
        }
        self.extras = {}
        self.last_score_reason = ["none"] * self.num_boards
        self.last_scored_by = torch.full((self.num_boards,), -1, dtype=torch.long, device=self.device)

        self.reset()

    def _side_indices(self, board_idx):
        return torch.cat([board_idx, board_idx + self.num_boards], dim=0)

    def _set_pose(self, entity, xy, z, envs_idx, *, zero_velocity=True):
        pos = torch.zeros((len(envs_idx), 3), dtype=gs.tc_float, device=self.device)
        pos[:, :2] = xy
        pos[:, 2] = z
        quat = self.identity_quat.repeat(len(envs_idx), 1)
        entity.set_pos(pos, envs_idx=envs_idx, zero_velocity=zero_velocity)
        entity.set_quat(quat, envs_idx=envs_idx, zero_velocity=False)
        entity.set_dofs_velocity(
            torch.zeros((len(envs_idx), entity.n_dofs), dtype=gs.tc_float, device=self.device),
            envs_idx=envs_idx,
        )

    def _reset_boards(self, board_idx, *, clear_reset_buf, clear_score_info=True):
        if len(board_idx) == 0:
            return

        n_boards = len(board_idx)
        left_xy = self.left_start.repeat(n_boards, 1)
        right_xy = self.right_start.repeat(n_boards, 1)
        left_xy[:, 1] += gs_rand_float(-0.025, 0.025, (n_boards,), self.device)
        right_xy[:, 1] += gs_rand_float(-0.025, 0.025, (n_boards,), self.device)

        self._set_pose(self.left, left_xy, STRIKER_Z, board_idx)
        self._set_pose(self.right, right_xy, STRIKER_Z, board_idx)

        side = torch.where(
            torch.rand((n_boards,), device=self.device) < 0.5,
            -torch.ones((n_boards,), dtype=gs.tc_float, device=self.device),
            torch.ones((n_boards,), dtype=gs.tc_float, device=self.device),
        )
        ball_xy = torch.zeros((n_boards, 2), dtype=gs.tc_float, device=self.device)
        ball_xy[:, 0] = side * gs_rand_float(BALL_START_MIN_X, BALL_START_MAX_X, (n_boards,), self.device)
        ball_xy[:, 1] = gs_rand_float(-BALL_START_Y_RANGE, BALL_START_Y_RANGE, (n_boards,), self.device)
        self._set_pose(self.ball, ball_xy, BALL_Z, board_idx)
        ball_vel = torch.zeros((n_boards, self.ball.n_dofs), dtype=gs.tc_float, device=self.device)
        ball_vel[:, 0] = gs_rand_float(-0.08, 0.08, (n_boards,), self.device)
        ball_vel[:, 1] = gs_rand_float(-0.08, 0.08, (n_boards,), self.device)
        self.ball.set_dofs_velocity(ball_vel, envs_idx=board_idx)

        for biscuit_idx, biscuit in enumerate(self.biscuits):
            xy = torch.zeros((n_boards, 2), dtype=gs.tc_float, device=self.device)
            xy[:, 1] = self.biscuit_start_y[biscuit_idx]
            self._set_pose(biscuit, xy, BISCUIT_Z, board_idx)

        self.biscuit_attached_to[board_idx] = -1
        self.biscuit_contact_frames[board_idx] = 0
        self.biscuit_offsets[board_idx] = 0
        self.left_action[board_idx] = 0
        self.right_action[board_idx] = 0
        self.last_ball_toucher[board_idx] = -1
        self.new_biscuit_attachments[board_idx] = 0

        side_idx = self._side_indices(board_idx)
        self.episode_length_buf[side_idx] = 0
        self.last_actions[side_idx] = 0
        for value in self.episode_sums.values():
            value[side_idx] = 0
        if clear_reset_buf:
            self.reset_buf[side_idx] = True
        if clear_score_info:
            reset_ids = set(board_idx.detach().cpu().tolist())
            self.last_score_reason = [
                "none" if i in reset_ids else reason for i, reason in enumerate(self.last_score_reason)
            ]
            self.last_scored_by[board_idx] = -1

    def _body_xy(self, entity):
        return entity.get_pos()[:, :2]

    def _body_vel_xy(self, entity):
        return entity.get_dofs_velocity()[:, :2]

    def _apply_striker_actions(self, actions):
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        self.left_action = self.actions[: self.num_boards]
        right_canonical = self.actions[self.num_boards :]
        self.right_action = torch.stack([-right_canonical[:, 0], right_canonical[:, 1]], dim=1)

        left_vel = torch.clip(self.left_action * self.handle_speed, -MAX_HANDLE_VEL, MAX_HANDLE_VEL)
        right_vel = torch.clip(self.right_action * self.handle_speed, -MAX_HANDLE_VEL, MAX_HANDLE_VEL)
        left_vel = self._stop_at_bounds(self.left, left_vel, "left")
        right_vel = self._stop_at_bounds(self.right, right_vel, "right")

        self._set_body_velocity(self.left, left_vel)
        self._set_body_velocity(self.right, right_vel)

    def _set_body_velocity(self, entity, vel_xy):
        vel = torch.zeros((self.num_boards, entity.n_dofs), dtype=gs.tc_float, device=self.device)
        vel[:, :2] = vel_xy
        entity.set_dofs_velocity(vel)

    def _stop_at_bounds(self, entity, vel_xy, side):
        pos = entity.get_pos()
        x_min, x_max = striker_x_bounds(side)
        y_min, y_max = striker_y_bounds()
        vel = vel_xy.clone()
        vel[:, 0] = torch.where((pos[:, 0] <= x_min) & (vel[:, 0] < 0.0), 0.0, vel[:, 0])
        vel[:, 0] = torch.where((pos[:, 0] >= x_max) & (vel[:, 0] > 0.0), 0.0, vel[:, 0])
        vel[:, 1] = torch.where((pos[:, 1] <= y_min) & (vel[:, 1] < 0.0), 0.0, vel[:, 1])
        vel[:, 1] = torch.where((pos[:, 1] >= y_max) & (vel[:, 1] > 0.0), 0.0, vel[:, 1])
        return vel

    def _constrain_striker(self, entity, side):
        pos = entity.get_pos()
        x_min, x_max = striker_x_bounds(side)
        y_min, y_max = striker_y_bounds()
        pos = torch.stack(
            [
                torch.clamp(pos[:, 0], x_min, x_max),
                torch.clamp(pos[:, 1], y_min, y_max),
                torch.full((self.num_boards,), STRIKER_Z, dtype=gs.tc_float, device=self.device),
            ],
            dim=1,
        )
        entity.set_pos(pos, zero_velocity=False)
        entity.set_quat(self.identity_quat.repeat(self.num_boards, 1), zero_velocity=False)
        vel = entity.get_dofs_velocity()
        speed = torch.norm(vel[:, :2], dim=1)
        scale = torch.where(speed > MAX_HANDLE_VEL, MAX_HANDLE_VEL / torch.clamp(speed, min=1e-6), 1.0)
        vel[:, :2] *= scale[:, None]
        vel[:, 2:] = 0.0
        entity.set_dofs_velocity(vel)

    def _constrain_disc(self, entity, z, max_speed, *, upright=True):
        pos = entity.get_pos()
        pos[:, 2] = z
        entity.set_pos(pos, zero_velocity=False)
        if upright:
            entity.set_quat(self.identity_quat.repeat(self.num_boards, 1), zero_velocity=False)

        vel = entity.get_dofs_velocity()
        speed = torch.norm(vel[:, :2], dim=1)
        scale = torch.where(speed > max_speed, max_speed / torch.clamp(speed, min=1e-6), 1.0)
        vel[:, :2] *= scale[:, None]
        vel[:, 2] = 0.0
        if upright:
            vel[:, 3:] = 0.0
        else:
            angular_speed = torch.norm(vel[:, 3:], dim=1)
            angular_max = max_speed / BALL_RADIUS
            angular_scale = torch.where(
                angular_speed > angular_max,
                angular_max / torch.clamp(angular_speed, min=1e-6),
                1.0,
            )
            vel[:, 3:] *= angular_scale[:, None]
        entity.set_dofs_velocity(vel)

    def _pin_attached_biscuits(self):
        left_xy = self._body_xy(self.left)
        right_xy = self._body_xy(self.right)
        left_vel = self._body_vel_xy(self.left)
        right_vel = self._body_vel_xy(self.right)
        x_min, x_max, y_min, y_max = biscuit_bounds()

        for biscuit_idx, biscuit in enumerate(self.biscuits):
            owner = self.biscuit_attached_to[:, biscuit_idx]
            for owner_id, striker_xy, striker_vel in ((0, left_xy, left_vel), (1, right_xy, right_vel)):
                envs_idx = (owner == owner_id).nonzero(as_tuple=False).reshape((-1,))
                if len(envs_idx) == 0:
                    continue
                xy = striker_xy[envs_idx] + self.biscuit_offsets[envs_idx, biscuit_idx]
                xy[:, 0] = torch.clamp(xy[:, 0], x_min, x_max)
                xy[:, 1] = torch.clamp(xy[:, 1], y_min, y_max)
                pos = torch.zeros((len(envs_idx), 3), dtype=gs.tc_float, device=self.device)
                pos[:, :2] = xy
                pos[:, 2] = BISCUIT_Z
                biscuit.set_pos(pos, envs_idx=envs_idx, zero_velocity=False)
                biscuit.set_quat(self.identity_quat.repeat(len(envs_idx), 1), envs_idx=envs_idx, zero_velocity=False)
                vel = torch.zeros((len(envs_idx), biscuit.n_dofs), dtype=gs.tc_float, device=self.device)
                vel[:, :2] = striker_vel[envs_idx]
                biscuit.set_dofs_velocity(vel, envs_idx=envs_idx)

    def _update_biscuit_attachment_state(self):
        self.new_biscuit_attachments.zero_()
        left_xy = self._body_xy(self.left)
        right_xy = self._body_xy(self.right)
        fallback_left = torch.tensor((-1.0, 0.0), dtype=gs.tc_float, device=self.device)
        fallback_right = torch.tensor((1.0, 0.0), dtype=gs.tc_float, device=self.device)

        for biscuit_idx, biscuit in enumerate(self.biscuits):
            owner = self.biscuit_attached_to[:, biscuit_idx]
            free = owner == -1
            biscuit_xy = self._body_xy(biscuit)
            left_delta = biscuit_xy - left_xy
            right_delta = biscuit_xy - right_xy
            left_dist = torch.norm(left_delta, dim=1)
            right_dist = torch.norm(right_delta, dim=1)
            left_contact = free & (left_dist <= self.attach_distance)
            right_contact = free & (right_dist <= self.attach_distance)

            self.biscuit_contact_frames[:, biscuit_idx, 0] = torch.where(
                left_contact,
                self.biscuit_contact_frames[:, biscuit_idx, 0] + 1,
                torch.zeros_like(self.biscuit_contact_frames[:, biscuit_idx, 0]),
            )
            self.biscuit_contact_frames[:, biscuit_idx, 1] = torch.where(
                right_contact,
                self.biscuit_contact_frames[:, biscuit_idx, 1] + 1,
                torch.zeros_like(self.biscuit_contact_frames[:, biscuit_idx, 1]),
            )

            attach_left = free & (self.biscuit_contact_frames[:, biscuit_idx, 0] >= self.attach_frames)
            attach_right = free & (self.biscuit_contact_frames[:, biscuit_idx, 1] >= self.attach_frames)
            attach_left = attach_left & (~attach_right | (left_dist <= right_dist))
            attach_right = attach_right & (~attach_left | (right_dist < left_dist))

            self._attach_biscuit(biscuit_idx, attach_left, 0, left_delta, fallback_left)
            self._attach_biscuit(biscuit_idx, attach_right, 1, right_delta, fallback_right)

    def _attach_biscuit(self, biscuit_idx, mask, owner_id, delta, fallback):
        envs_idx = mask.nonzero(as_tuple=False).reshape((-1,))
        if len(envs_idx) == 0:
            return
        self.biscuit_attached_to[envs_idx, biscuit_idx] = owner_id
        self.new_biscuit_attachments[envs_idx, owner_id] += 1.0
        direction = delta[envs_idx]
        norm = torch.norm(direction, dim=1, keepdim=True)
        direction = torch.where(norm > 1e-6, direction / torch.clamp(norm, min=1e-6), fallback.repeat(len(envs_idx), 1))
        self.biscuit_offsets[envs_idx, biscuit_idx] = direction * self.attach_distance

    def _contacts_with_ball(self):
        ball_xy = self._body_xy(self.ball)
        contact_distance = STRIKER_RADIUS + BALL_RADIUS + 0.003
        left_dist = torch.norm(ball_xy - self._body_xy(self.left), dim=1)
        right_dist = torch.norm(ball_xy - self._body_xy(self.right), dim=1)
        self.left_contact = left_dist <= contact_distance
        self.right_contact = right_dist <= contact_distance

        touched_by = torch.full((self.num_boards,), -1, dtype=torch.long, device=self.device)
        touched_by = torch.where(self.left_contact, torch.zeros_like(touched_by), touched_by)
        right_is_clearer = ~self.left_contact | (right_dist < left_dist)
        touched_by = torch.where(self.right_contact & right_is_clearer, torch.ones_like(touched_by), touched_by)
        self.last_ball_toucher = torch.where(touched_by >= 0, touched_by, self.last_ball_toucher)

    def _detect_scores(self):
        ball_pos = self.ball.get_pos()
        half_w = BOARD_WIDTH / 2.0
        half_h = BOARD_HEIGHT / 2.0
        goal_half = GOAL_WIDTH / 2.0
        left_goal = (ball_pos[:, 0] > half_w + BALL_RADIUS) & (torch.abs(ball_pos[:, 1]) <= goal_half)
        right_goal = (ball_pos[:, 0] < -half_w - BALL_RADIUS) & (torch.abs(ball_pos[:, 1]) <= goal_half)
        off_right = ball_pos[:, 0] > half_w + BALL_RADIUS
        off_left = ball_pos[:, 0] < -half_w - BALL_RADIUS
        off_y = torch.abs(ball_pos[:, 1]) > half_h + BALL_RADIUS
        out_of_bounds = off_y | (off_right & ~left_goal) | (off_left & ~right_goal)

        left_biscuits = torch.sum(self.biscuit_attached_to == 0, dim=1) >= self.biscuit_score_threshold
        right_biscuits = torch.sum(self.biscuit_attached_to == 1, dim=1) >= self.biscuit_score_threshold

        scored_by = torch.full((self.num_boards,), -1, dtype=torch.long, device=self.device)
        score_reason = ["none"] * self.num_boards
        scored_by = torch.where(left_goal | right_biscuits, torch.zeros_like(scored_by), scored_by)
        scored_by = torch.where((right_goal | left_biscuits) & (scored_by == -1), torch.ones_like(scored_by), scored_by)

        fallback_offender = torch.where(ball_pos[:, 0] >= 0.0, torch.zeros_like(scored_by), torch.ones_like(scored_by))
        offender = torch.where(self.last_ball_toucher >= 0, self.last_ball_toucher, fallback_offender)
        out_scored_by = 1 - offender
        scored_by = torch.where(out_of_bounds & (scored_by == -1), out_scored_by, scored_by)

        for idx in (left_goal | right_goal).nonzero(as_tuple=False).reshape((-1,)).detach().cpu().tolist():
            score_reason[idx] = "goal"
        for idx in (left_biscuits | right_biscuits).nonzero(as_tuple=False).reshape((-1,)).detach().cpu().tolist():
            if score_reason[idx] == "none":
                score_reason[idx] = "biscuits"
        for idx in out_of_bounds.nonzero(as_tuple=False).reshape((-1,)).detach().cpu().tolist():
            if score_reason[idx] == "none":
                score_reason[idx] = "out_of_bounds"
        self.last_score_reason = score_reason
        self.last_scored_by = scored_by.clone()
        return scored_by, out_of_bounds

    def _canonical_state(self, side):
        sign = 1.0 if side == "left" else -1.0
        own = self.left if side == "left" else self.right
        opponent = self.right if side == "left" else self.left
        own_owner_id = 0 if side == "left" else 1
        opp_owner_id = 1 - own_owner_id

        own_pos = own.get_pos()
        own_vel = self._body_vel_xy(own)
        opp_pos = opponent.get_pos()
        opp_vel = self._body_vel_xy(opponent)
        ball_pos = self.ball.get_pos()
        ball_vel = self._body_vel_xy(self.ball)

        half_w = BOARD_WIDTH / 2.0
        half_h = BOARD_HEIGHT / 2.0
        time_remaining = 1.0 - torch.clamp(
            self.episode_length_buf[: self.num_boards].to(gs.tc_float) / self.max_episode_length,
            0.0,
            1.0,
        )

        base = [
            sign * own_pos[:, 0] / half_w,
            own_pos[:, 1] / half_h,
            sign * own_vel[:, 0] / self.handle_speed,
            own_vel[:, 1] / self.handle_speed,
            sign * opp_pos[:, 0] / half_w,
            opp_pos[:, 1] / half_h,
            sign * opp_vel[:, 0] / self.handle_speed,
            opp_vel[:, 1] / self.handle_speed,
            sign * ball_pos[:, 0] / half_w,
            ball_pos[:, 1] / half_h,
            sign * ball_vel[:, 0] / self.ball_max_speed,
            ball_vel[:, 1] / self.ball_max_speed,
            sign * (ball_pos[:, 0] - own_pos[:, 0]) / BOARD_WIDTH,
            (ball_pos[:, 1] - own_pos[:, 1]) / BOARD_HEIGHT,
            torch.zeros((self.num_boards,), dtype=gs.tc_float, device=self.device),
            time_remaining,
        ]

        biscuit_features = []
        for biscuit_idx, biscuit in enumerate(self.biscuits):
            biscuit_pos = biscuit.get_pos()
            biscuit_vel = self._body_vel_xy(biscuit)
            owner = self.biscuit_attached_to[:, biscuit_idx]
            attached_state = torch.zeros((self.num_boards,), dtype=gs.tc_float, device=self.device)
            attached_state = torch.where(owner == own_owner_id, torch.ones_like(attached_state), attached_state)
            attached_state = torch.where(owner == opp_owner_id, -torch.ones_like(attached_state), attached_state)
            biscuit_features.extend(
                [
                    sign * biscuit_pos[:, 0] / half_w,
                    biscuit_pos[:, 1] / half_h,
                    sign * biscuit_vel[:, 0] / self.ball_max_speed,
                    biscuit_vel[:, 1] / self.ball_max_speed,
                    attached_state,
                ]
            )

        own_count = torch.sum(self.biscuit_attached_to == own_owner_id, dim=1).to(gs.tc_float)
        opp_count = torch.sum(self.biscuit_attached_to == opp_owner_id, dim=1).to(gs.tc_float)
        return torch.stack(
            [
                *base,
                *biscuit_features,
                own_count / self.biscuit_score_threshold,
                opp_count / self.biscuit_score_threshold,
            ],
            dim=1,
        )

    def _update_observation(self):
        left_obs = self._canonical_state("left")
        right_obs = self._canonical_state("right")
        self.obs_buf = torch.clip(torch.cat([left_obs, right_obs], dim=0), -1.0, 1.0)

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _biscuit_proximity_risk(self, side):
        striker = self.left if side == "left" else self.right
        owner_id = 0 if side == "left" else 1
        striker_xy = self._body_xy(striker)
        risk = torch.zeros((self.num_boards,), dtype=gs.tc_float, device=self.device)
        attached = torch.sum(self.biscuit_attached_to == owner_id, dim=1).to(gs.tc_float)
        for biscuit_idx, biscuit in enumerate(self.biscuits):
            free = self.biscuit_attached_to[:, biscuit_idx] == -1
            distance = torch.norm(self._body_xy(biscuit) - striker_xy, dim=1)
            proximity = 1.0 - distance / self.biscuit_attraction_range
            risk += torch.where(
                free & (distance < self.biscuit_attraction_range), torch.clamp(proximity, 0.0, 1.0), 0.0
            )
        return attached, risk

    def _side_reward_components(self, side, previous_ball_x, action, contact):
        obs = self._canonical_state(side)
        side_idx = 0 if side == "left" else 1
        ball_x = obs[:, 8]
        ball_y = obs[:, 9]
        ball_vx = obs[:, 10]
        own_y = obs[:, 1]
        ball_distance = torch.norm(obs[:, 12:14], dim=1)

        defensive_need = torch.clamp(-ball_x, min=0.0)
        y_alignment = 1.0 - torch.clamp(torch.abs(own_y - ball_y), 0.0, 1.0)
        attached, proximity = self._biscuit_proximity_risk(side)
        return {
            "progress": self.reward_cfg["progress"] * (ball_x - previous_ball_x),
            "position": self.reward_cfg["puck_position"] * ball_x,
            "speed": self.reward_cfg["puck_speed"] * ball_vx,
            "contact": self.reward_cfg["contact"] * contact.to(gs.tc_float),
            "distance": self.reward_cfg["puck_distance"] * (1.0 - torch.clamp(ball_distance * 2.0, 0.0, 1.0)),
            "defense": self.reward_cfg["defense"] * defensive_need * y_alignment,
            "danger": -self.reward_cfg["own_goal_danger"] * defensive_need * (1.0 - torch.abs(ball_y)),
            "time": -torch.full(
                (self.num_boards,), self.reward_cfg["time_penalty"], dtype=gs.tc_float, device=self.device
            ),
            "action": -self.reward_cfg["action_penalty"] * torch.sum(action * action, dim=1),
            "biscuit_attach": -(
                self.reward_cfg["biscuit_attached_penalty"] * attached
                + self.reward_cfg["biscuit_attach_penalty"] * self.new_biscuit_attachments[:, side_idx]
            ),
            "biscuit_pull": -self.reward_cfg["biscuit_proximity_penalty"] * proximity,
            "terminal": torch.zeros((self.num_boards,), dtype=gs.tc_float, device=self.device),
        }

    def _compute_reward(self, previous_left_ball_x, previous_right_ball_x, scored_by, out_of_bounds):
        left_components = self._side_reward_components(
            "left",
            previous_left_ball_x,
            self.actions[: self.num_boards],
            self.left_contact,
        )
        right_components = self._side_reward_components(
            "right",
            previous_right_ball_x,
            self.actions[self.num_boards :],
            self.right_contact,
        )

        rewards = {}
        for name in self.reward_names:
            if name == "terminal":
                terminal_value = torch.where(
                    out_of_bounds,
                    torch.full(
                        (self.num_boards,),
                        self.reward_cfg["out_of_bounds_penalty"],
                        dtype=gs.tc_float,
                        device=self.device,
                    ),
                    torch.full(
                        (self.num_boards,), self.reward_cfg["terminal_goal"], dtype=gs.tc_float, device=self.device
                    ),
                )
                left_terminal = torch.where(
                    scored_by == 0,
                    terminal_value,
                    torch.where(scored_by == 1, -terminal_value, 0.0),
                )
                right_terminal = -left_terminal
                rewards[name] = torch.cat([left_terminal, right_terminal], dim=0)
            else:
                rewards[name] = torch.cat(
                    [left_components[name] - right_components[name], right_components[name] - left_components[name]],
                    dim=0,
                )

        self.rew_buf[:] = 0.0
        for name, reward in rewards.items():
            self.rew_buf += reward
            self.episode_sums[name] += reward

    def step(self, actions):
        ball_pos = self.ball.get_pos()
        half_w = BOARD_WIDTH / 2.0
        previous_left_ball_x = ball_pos[:, 0] / half_w
        previous_right_ball_x = -ball_pos[:, 0] / half_w

        self._apply_striker_actions(actions)
        for _ in range(self.frame_skip):
            self._constrain_striker(self.left, "left")
            self._constrain_striker(self.right, "right")
            self._constrain_disc(self.ball, BALL_Z, self.ball_max_speed, upright=False)
            for biscuit in self.biscuits:
                self._constrain_disc(biscuit, BISCUIT_Z, self.biscuit_max_speed)
            self._pin_attached_biscuits()
            self.scene.step()
            self._constrain_striker(self.left, "left")
            self._constrain_striker(self.right, "right")
            self._pin_attached_biscuits()
            self._constrain_disc(self.ball, BALL_Z, self.ball_max_speed, upright=False)
            for biscuit in self.biscuits:
                self._constrain_disc(biscuit, BISCUIT_Z, self.biscuit_max_speed)

        self.episode_length_buf += 1
        self._contacts_with_ball()
        self._update_biscuit_attachment_state()
        self._pin_attached_biscuits()
        scored_by, out_of_bounds = self._detect_scores()

        board_done = scored_by >= 0
        board_timeout = self.episode_length_buf[: self.num_boards] >= self.max_episode_length
        board_reset = board_done | board_timeout
        self.reset_buf = torch.cat([board_reset, board_reset], dim=0).to(gs.tc_int)
        self.extras["time_outs"] = torch.cat([board_timeout, board_timeout], dim=0).to(gs.tc_float)

        self._compute_reward(previous_left_ball_x, previous_right_ball_x, scored_by, out_of_bounds)
        done_boards = board_reset.nonzero(as_tuple=False).reshape((-1,))
        self.extras["episode"] = {}
        if len(done_boards) > 0:
            done_sides = self._side_indices(done_boards)
            for name, value in self.episode_sums.items():
                self.extras["episode"]["rew_" + name] = torch.mean(value[done_sides]).item()
            self.extras["episode"]["score_rate"] = torch.mean(board_done[done_boards].to(gs.tc_float)).item()
            self._reset_boards(done_boards, clear_reset_buf=False, clear_score_info=False)

        self._update_observation()
        self.last_actions[:] = self.actions[:]
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        all_boards = torch.arange(self.num_boards, dtype=torch.long, device=self.device)
        self._reset_boards(all_boards, clear_reset_buf=True)
        self._update_observation()
        return self.get_observations()


def get_default_env_cfg(num_boards):
    sim_dt = 1 / 480
    frame_skip = 8
    return {
        "num_boards": num_boards,
        "num_actions": 2,
        "episode_length_s": 15.0,
        "sim_dt": sim_dt,
        "sim_substeps": 4,
        # constraint_timeconst must stay comfortably above 2*sim_dt or stiff striker/ball
        # contacts can produce NaN constraint forces. 2*sim_dt ~= 0.0042, so 0.004 sat right
        # at the stability edge and blew up once the trained policy hit harder; 0.01 (~5*dt)
        # gives a safe margin.
        "constraint_timeconst": 0.01,
        "ctrl_dt": sim_dt * frame_skip,
        "frame_skip": frame_skip,
        # The playground uses a much faster manual-control speed. PPO starts with
        # near-random saturated actions, so training uses the stabler KLASK-2 speed.
        "handle_speed": 1.8,
        "ball_max_speed": BALL_MAX_SPEED,
        "biscuit_max_speed": BISCUIT_MAX_SPEED,
        "clip_actions": 1.0,
        "biscuit_attach_frames": 2,
        "biscuit_attach_margin": 0.0015,
        "biscuit_score_threshold": 2,
        "biscuit_attraction_range": 0.044,
        "max_collision_pairs": 512,
        "rendered_envs": min(4, num_boards),
        "show_fps": False,
    }


def get_default_reward_cfg():
    return {
        "terminal_goal": 12.0,
        "out_of_bounds_penalty": 4.0,
        "progress": 0.6,
        "puck_position": 0.02,
        "puck_speed": 0.08,
        "contact": 0.08,
        "puck_distance": 0.025,
        "defense": 0.04,
        "own_goal_danger": 0.05,
        "biscuit_attached_penalty": 0.05,
        "biscuit_attach_penalty": 0.0,
        "biscuit_proximity_penalty": 0.01,
        "time_penalty": 0.0005,
        "action_penalty": 0.0003,
    }
