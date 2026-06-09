import math
from functools import partial

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.options.sensors import BatchRendererCameraOptions, RasterizerCameraOptions
from genesis.vis.camera import Camera
from genesis.utils.geom import (
    transform_quat_by_quat,
    transform_by_trans_quat,
)

try:
    import gs_madrona

    _ENABLE_MADRONA = True
except ImportError:
    _ENABLE_MADRONA = False


class GraspEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        # camera config
        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.record_scene_style = env_cfg.get("record_scene_style", "plain")

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(10, self.num_envs))),
                env_separate_rigid=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=(2.5, -1.0, 2.5),
                camera_lookat=(0.5, -0.3, 0.1),
                camera_fov=55,
                max_FPS=int(0.5 / self.ctrl_dt),
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # == add ground ==
        ground_morph = (
            gs.morphs.Plane(visualization=False, collision=True)
            if self.record_scene_style == "studio"
            else gs.morphs.Plane()
        )
        self.scene.add_entity(
            ground_morph,
            # gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        )

        # == add robot ==
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        # == add object ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["box_size"],
                fixed=env_cfg.get("box_fixed", True),
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Plastic(
                color=env_cfg.get("box_color", (0.95, 0.05, 0.02)),
                roughness=0.55,
            ),
        )

        if self.record_scene_style == "studio":
            self._add_studio_scene()

        # == visualization camera (debug only, uses scene camera API) ==
        if self.env_cfg.get("visualize_camera", False):
            self.vis_cam = self.scene.add_camera(
                res=(1280, 960),
                pos=(3.5, 0.0, 2.5),
                lookat=(1.2, 1.0, 0.0),
                fov=52,
                GUI=False,
                debug=True,
            )

        def _build_nyx_camera_options(res, pos, lookat, fov):
            try:
                import gs_nyx.nyx_py_renderer as npr
                import gs_nyx.nyx_py_sdk as nps
                from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
            except ImportError as exc:
                raise ImportError(
                    "Nyx rendering requires `gs-nyx-plugin`; run with `uv run --with gs-nyx-plugin`."
                ) from exc

            nyx_cfg = self.env_cfg.get("nyx", {})
            env_maps = []
            if nyx_cfg.get("env_map"):
                env_map = nps.EnvironmentMapAsset()
                env_map.texture = nyx_cfg["env_map"]
                env_map.layout = nps.EEnvMapLayout.LongLat
                env_map.multiplier = nyx_cfg.get("env_map_multiplier", 2.0)
                env_maps.append(env_map)

            render_mode = getattr(npr.ERenderMode, nyx_cfg.get("render_mode", "FastPathTracer"))
            lights = nyx_cfg.get(
                "lights",
                [
                    {
                        "type": "directional",
                        "dir": (-0.45, -0.25, -0.85),
                        "color": (1.0, 0.96, 0.9),
                        "intensity": 4.0,
                        "shadow": True,
                    }
                ],
            )

            return NyxCameraOptions(
                res=res,
                pos=pos,
                lookat=lookat,
                fov=fov,
                spp=nyx_cfg.get("spp", 32),
                render_mode=render_mode,
                lights=lights,
                env_maps=env_maps,
            )

        # == stereo camera sensors (lazy rendering — zero cost until read()) ==
        default_policy_cameras = {
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
        }
        policy_cameras = env_cfg.get("policy_cameras", {})
        left_policy_camera = {**default_policy_cameras["left_cam"], **policy_cameras.get("left_cam", {})}
        right_policy_camera = {**default_policy_cameras["right_cam"], **policy_cameras.get("right_cam", {})}

        policy_renderer = env_cfg.get("policy_renderer", "auto")
        if policy_renderer == "nyx":
            left_cam_options = _build_nyx_camera_options(
                res=(self.image_width, self.image_height),
                pos=left_policy_camera["pos"],
                lookat=left_policy_camera["lookat"],
                fov=left_policy_camera.get("fov", 55),
            )
            right_cam_options = _build_nyx_camera_options(
                res=(self.image_width, self.image_height),
                pos=right_policy_camera["pos"],
                lookat=right_policy_camera["lookat"],
                fov=right_policy_camera.get("fov", 55),
            )
        elif policy_renderer == "rasterizer":
            left_cam_options = RasterizerCameraOptions(
                res=(self.image_width, self.image_height),
                pos=left_policy_camera["pos"],
                lookat=left_policy_camera["lookat"],
                fov=left_policy_camera.get("fov", 55),
            )
            right_cam_options = RasterizerCameraOptions(
                res=(self.image_width, self.image_height),
                pos=right_policy_camera["pos"],
                lookat=right_policy_camera["lookat"],
                fov=right_policy_camera.get("fov", 55),
            )
        elif policy_renderer == "auto" and _ENABLE_MADRONA and gs.backend == gs.cuda:
            left_cam_options = BatchRendererCameraOptions(
                res=(self.image_width, self.image_height),
                pos=left_policy_camera["pos"],
                lookat=left_policy_camera["lookat"],
                fov=left_policy_camera.get("fov", 55),
                use_rasterizer=True,
            )
            right_cam_options = BatchRendererCameraOptions(
                res=(self.image_width, self.image_height),
                pos=right_policy_camera["pos"],
                lookat=right_policy_camera["lookat"],
                fov=right_policy_camera.get("fov", 55),
                use_rasterizer=True,
            )
        elif policy_renderer == "auto":
            left_cam_options = RasterizerCameraOptions(
                res=(self.image_width, self.image_height),
                pos=left_policy_camera["pos"],
                lookat=left_policy_camera["lookat"],
                fov=left_policy_camera.get("fov", 55),
            )
            right_cam_options = RasterizerCameraOptions(
                res=(self.image_width, self.image_height),
                pos=right_policy_camera["pos"],
                lookat=right_policy_camera["lookat"],
                fov=right_policy_camera.get("fov", 55),
            )
        else:
            raise ValueError(f"Unsupported policy renderer: {policy_renderer}")

        self.left_cam = self.scene.add_sensor(left_cam_options)
        self.right_cam = self.scene.add_sensor(right_cam_options)

        self._record_cameras = {}
        record_image_resolution = env_cfg.get("record_image_resolution")
        if record_image_resolution is not None:
            record_width, record_height = record_image_resolution
            record_cameras = env_cfg.get(
                "record_cameras",
                {
                    "record_left_cam": {
                        "pos": left_policy_camera["pos"],
                        "lookat": left_policy_camera["lookat"],
                        "fov": left_policy_camera.get("fov", 55),
                    },
                    "record_right_cam": {
                        "pos": right_policy_camera["pos"],
                        "lookat": right_policy_camera["lookat"],
                        "fov": right_policy_camera.get("fov", 55),
                    },
                },
            )
            record_renderer = env_cfg.get("record_renderer", "rasterizer")
            for cam_name, camera_cfg in record_cameras.items():
                camera_res = tuple(camera_cfg.get("res", (record_width, record_height)))
                camera_pos = camera_cfg["pos"]
                camera_lookat = camera_cfg["lookat"]
                camera_fov = camera_cfg.get("fov", 60)
                if record_renderer == "nyx":
                    camera_options = _build_nyx_camera_options(camera_res, camera_pos, camera_lookat, camera_fov)
                elif record_renderer == "rasterizer":
                    # Keep recording cameras off the batch renderer so they can use a
                    # different resolution from the policy input cameras.
                    camera_options = RasterizerCameraOptions(
                        res=camera_res,
                        pos=camera_pos,
                        lookat=camera_lookat,
                        fov=camera_fov,
                    )
                else:
                    raise ValueError(f"Unsupported record renderer: {record_renderer}")

                camera = self.scene.add_sensor(camera_options)
                setattr(self, cam_name, camera)
                self._record_cameras[cam_name] = camera

        # == camera data readers ==
        def _read_scene_cam(cam):
            rgb = cam.render(rgb=True)[0]
            if rgb.ndim == 4:
                rgb = rgb[0]
            return rgb[..., :3]

        def _read_sensor_cam(cam):
            rgb = cam.read().rgb
            if rgb.ndim == 4:
                rgb = rgb[0]
            return rgb

        # Debug live preview of sensor cameras
        if self.env_cfg.get("visualize_camera", False):
            self.scene.start_recording(
                data_func=partial(_read_sensor_cam, self.left_cam),
                rec_options=gs.recorders.MPLImagePlot(title="Left Camera"),
            )
            self.scene.start_recording(
                data_func=partial(_read_sensor_cam, self.right_cam),
                rec_options=gs.recorders.MPLImagePlot(title="Right Camera"),
            )

        # == set up video recording (must be before build) ==

        record_video = env_cfg.get("record_video", {})
        for cam_name, filename in record_video.items():
            cam = getattr(self, cam_name)
            reader = _read_scene_cam if isinstance(cam, Camera) else _read_sensor_cam
            self.scene.start_recording(
                data_func=partial(reader, cam),
                rec_options=gs.recorders.VideoFile(filename=filename),
            )

        # build
        self.scene.build(n_envs=env_cfg["num_envs"], env_spacing=(1.0, 1.0))
        # set pd gains (must be called after scene.build)
        self.robot.set_pd_gains()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs,
            device=self.device,
            unit_length=self.env_cfg.get("keypoint_unit_length", 0.04),
        )
        # == init buffers ==
        self._init_buffers()
        self.reset()

    def _add_studio_scene(self) -> None:
        """Add visual-only room geometry for nicer Nyx recordings."""

        def surface(color, roughness=0.7):
            return gs.surfaces.Plastic(color=color, roughness=roughness)

        def add_box(name, pos, size, surface):
            self.scene.add_entity(
                gs.morphs.Box(
                    pos=pos,
                    size=size,
                    fixed=True,
                    visualization=True,
                    collision=False,
                    batch_fixed_verts=False,
                ),
                surface=surface,
                name=name,
            )

        table_surface = surface((0.47, 0.31, 0.18), roughness=0.48)
        mat_surface = surface((0.13, 0.14, 0.14), roughness=0.78)
        leg_surface = surface((0.25, 0.23, 0.21), roughness=0.52)
        wall_surface = surface((0.76, 0.74, 0.69), roughness=0.88)
        floor_surface = surface((0.43, 0.42, 0.39), roughness=0.82)
        trim_surface = surface((0.62, 0.59, 0.54), roughness=0.8)
        light_surface = gs.surfaces.Emission(emissive=(1.0, 0.92, 0.78))

        add_box(
            "studio_tabletop",
            pos=(0.0, -0.28, -0.05),
            size=(0.9, 0.75, 0.075),
            surface=table_surface,
        )
        add_box(
            "studio_work_mat",
            pos=(0.0, -0.28, -0.011),
            size=(0.48, 0.30, 0.01),
            surface=mat_surface,
        )
        for x_idx, x in enumerate((-0.36, 0.36)):
            for y_idx, y in enumerate((-0.58, 0.02)):
                add_box(
                    f"studio_table_leg_{x_idx}_{y_idx}",
                    pos=(x, y, -0.395),
                    size=(0.06, 0.06, 0.62),
                    surface=leg_surface,
                )

        add_box("studio_floor", pos=(0.0, -0.28, -0.73), size=(1.6, 1.6, 0.04), surface=floor_surface)
        add_box("studio_back_wall", pos=(0.0, 0.12, 0.05), size=(1.6, 0.04, 1.55), surface=wall_surface)
        add_box("studio_left_wall", pos=(-0.58, -0.28, 0.05), size=(0.04, 0.8, 1.55), surface=wall_surface)
        add_box(
            "studio_back_baseboard",
            pos=(0.0, 0.085, -0.585),
            size=(1.4, 0.035, 0.08),
            surface=trim_surface,
        )
        add_box(
            "studio_left_baseboard",
            pos=(-0.545, -0.28, -0.585),
            size=(0.035, 0.7, 0.08),
            surface=trim_surface,
        )
        add_box(
            "studio_light_panel",
            pos=(0.0, -0.18, 0.79),
            size=(0.42, 0.22, 0.02),
            surface=light_surface,
        )

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.ones(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device, dtype=gs.tc_float)
        self.extras = dict()

    def _reset_idx(self, envs_idx=None) -> None:
        """Reset specified environments.

        Parameters
        ----------
        envs_idx : torch.Tensor or None
            Boolean mask of shape (num_envs,) for selective reset, or None for full reset.
        """
        # Reset robot
        self.robot.reset(envs_idx)

        # Generate random object state for all envs inside the SO-100 reachable workspace.
        x_bounds = self.env_cfg.get("object_x_bounds", (-0.12, 0.12))
        y_bounds = self.env_cfg.get("object_y_bounds", (-0.32, -0.24))
        random_x = torch.rand(self.num_envs, device=self.device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
        random_y = torch.rand(self.num_envs, device=self.device) * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
        random_z = torch.full(
            (self.num_envs,),
            self.env_cfg.get("object_z", self.env_cfg["box_size"][2] / 2),
            device=self.device,
        )
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        random_yaw = (torch.rand(self.num_envs, device=self.device) * 2 * math.pi - math.pi) * 0.25
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)
        goal_pose = torch.cat([random_pos, goal_yaw], dim=-1)

        # Reset object — set_pos/set_quat with skip_forward, then FK runs once for everything
        if envs_idx is None:
            self.goal_pose.copy_(goal_pose)
            self.object.set_pos(random_pos, skip_forward=True)
            self.object.set_quat(goal_yaw, skip_forward=False)
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], goal_pose, self.goal_pose, out=self.goal_pose)
            self.object.set_pos(random_pos, envs_idx=envs_idx, skip_forward=True)
            self.object.set_quat(goal_yaw, envs_idx=envs_idx, skip_forward=False)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # Invalidate camera caches after state change
        self.left_cam._stale = True
        self.right_cam._stale = True
        for cam in self._record_cameras.values():
            if hasattr(cam, "_stale"):
                cam._stale = True

        # Fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

    def reset(self) -> TensorDict:
        self._reset_idx()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # apply action
        actions = self.rescale_action(actions)
        self.robot.apply_action(actions, open_gripper=True)
        self.scene.step()

        # update time
        self.episode_length_buf += 1

        # check termination (bool mask)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # timeout for value bootstrapping (only true timeouts, not NaN errors)
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # compute reward before reset (reflects terminal state)
        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # soft-reset envs that need it
        self._reset_idx(self.reset_buf)

        return self.get_observations(), reward, self.reset_buf, self.extras

    def get_observations(self) -> TensorDict:
        # Current end-effector pose
        finger_pos, finger_quat = (
            self.robot.center_finger_pose[:, :3],
            self.robot.center_finger_pose[:, 3:7],
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()
        obs_components = [
            finger_pos - obj_pos,  # 3D position difference
            finger_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        self.obs_buf = torch.cat(obs_components, dim=-1)
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scales

    def get_stereo_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        rgb_left = self.left_cam.read().rgb  # (B, H, W, 3) or (H, W, 3)
        rgb_right = self.right_cam.read().rgb  # (B, H, W, 3) or (H, W, 3)

        if rgb_left.ndim == 3:
            if self.num_envs != 1:
                raise RuntimeError("Unbatched camera output can only be used with num_envs=1.")
            rgb_left = rgb_left.unsqueeze(0)
            rgb_right = rgb_right.unsqueeze(0)

        # Convert to (B, 3, H, W) float
        rgb_left = rgb_left.permute(0, 3, 1, 2).float()
        rgb_right = rgb_right.permute(0, 3, 1, 2).float()

        if normalize:
            rgb_left = rgb_left / 255.0
            rgb_right = rgb_right / 255.0

        # Concatenate left and right rgb images along channel dimension: [B, 6, H, W]
        return torch.cat([rgb_left, rgb_right], dim=1)

    # ------------ begin reward functions----------------
    def _reward_keypoints(self) -> torch.Tensor:
        keypoints_offset = self.keypoints_offset
        finger_pos_keypoints = self._to_world_frame(
            self.robot.center_finger_pose[:, :3],
            self.robot.center_finger_pose[:, 3:7],
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(self.object.get_pos(), self.object.get_quat(), keypoints_offset)
        dist = torch.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    # ------------ end reward functions----------------

    @staticmethod
    def _to_world_frame(
        position: torch.Tensor,  # [B, 3]
        quaternion: torch.Tensor,  # [B, 4]
        keypoints_offset: torch.Tensor,  # [B, K, 3]
    ) -> torch.Tensor:
        return transform_by_trans_quat(keypoints_offset, position.unsqueeze(1), quaternion.unsqueeze(1))

    @staticmethod
    def get_keypoint_offsets(batch_size: int, device: str, unit_length: float = 0.5) -> torch.Tensor:
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at body center.
        """
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],  # origin
                    [-1.0, 0, 0],  # x-negative
                    [1.0, 0, 0],  # x-positive
                    [0, -1.0, 0],  # y-negative
                    [0, 1.0, 0],  # y-positive
                    [0, 0, -1.0],  # z-negative
                    [0, 0, 1.0],  # z-positive
                ],
                device=device,
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets[None].repeat((batch_size, 1, 1))

    def grasp_and_lift_demo(self) -> None:
        total_steps = 500
        goal_pose = self.robot.ee_pose.clone()
        # lift pose (above the object)
        lift_height = self.env_cfg.get("scripted_lift_height", 0.10)
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        for i in range(total_steps):
            if i < total_steps / 4:  # grasping
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps / 2:  # lifting
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 3 / 4:  # hold lifted object
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            else:  # reset
                self.robot.go_home(open_gripper=True)
            self.scene.step()


## ------------ robot ----------------
class Manipulator:
    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.MJCF = gs.morphs.MJCF(
            file=args["mjcf_file"],
            pos=args.get("base_pos", (0.0, 0.0, 0.0)),
            quat=args.get("base_quat", (1.0, 0.0, 0.0, 0.0)),
        )
        self._robot_entity: gs.Entity = scene.add_entity(material=material, morph=morph)

        self._gripper_open_dof = torch.tensor(args["gripper_open_dof"], dtype=torch.float32, device=self._device)
        self._gripper_close_dof = torch.tensor(args["gripper_close_dof"], dtype=torch.float32, device=self._device)

        # == some buffer initialization ==
        self._init()

    def set_pd_gains(self):
        if "dof_kp" in self._args:
            self._robot_entity.set_dofs_kp(torch.tensor(self._args["dof_kp"], dtype=torch.float32))
        if "dof_kv" in self._args:
            self._robot_entity.set_dofs_kv(torch.tensor(self._args["dof_kv"], dtype=torch.float32))
        if "dof_force_lower" in self._args and "dof_force_upper" in self._args:
            self._robot_entity.set_dofs_force_range(
                torch.tensor(self._args["dof_force_lower"], dtype=torch.float32),
                torch.tensor(self._args["dof_force_upper"], dtype=torch.float32),
            )

    def _init(self):
        self._arm_joint_names = self._args["arm_joint_names"]
        self._gripper_joint_names = self._args["gripper_joint_names"]
        self._arm_dof_dim = len(self._arm_joint_names)
        self._gripper_dim = len(self._gripper_joint_names)

        self._arm_dof_idx = [self._robot_entity.get_joint(name).dofs_idx_local[0] for name in self._arm_joint_names]
        self._fingers_dof = [
            self._robot_entity.get_joint(name).dofs_idx_local[0] for name in self._gripper_joint_names
        ]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        self._default_joint_angles = list(self._args["default_arm_dof"])
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += list(self._args["default_gripper_dof"])
        self._init_qpos = torch.tensor(self._default_joint_angles, dtype=torch.float32, device=self._device)
        self._arm_lower_limits = torch.tensor(self._args["arm_lower_limits"], dtype=torch.float32, device=self._device)
        self._arm_upper_limits = torch.tensor(self._args["arm_upper_limits"], dtype=torch.float32, device=self._device)
        self._fixed_tip_offset = torch.tensor(
            self._args.get("fixed_finger_tip_offset", (0.012, -0.08, 0.0)),
            dtype=torch.float32,
            device=self._device,
        ).repeat(self._num_envs, 1)
        self._moving_tip_offset = torch.tensor(
            self._args.get("moving_finger_tip_offset", (-0.009, -0.055, 0.0)),
            dtype=torch.float32,
            device=self._device,
        ).repeat(self._num_envs, 1)

    def reset(self, envs_idx=None, skip_forward=True):
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )

    def apply_action(self, action: torch.Tensor, open_gripper: bool) -> None:
        """Apply the action to the robot."""
        q_pos = self._robot_entity.get_qpos()
        target_arm_qpos = q_pos[:, self._arm_dof_idx] + action
        q_pos[:, self._arm_dof_idx] = torch.clamp(target_arm_qpos, self._arm_lower_limits, self._arm_upper_limits)
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True):
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    def go_home(self, open_gripper: bool = True):
        q_pos = self._init_qpos.repeat(self._num_envs, 1)
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        """
        The end-effector pose (the hand pose)
        """
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        pos = transform_by_trans_quat(self._fixed_tip_offset, pos, quat)
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        pos, quat = (
            self._right_finger_link.get_pos(),
            self._right_finger_link.get_quat(),
        )
        pos = transform_by_trans_quat(self._moving_tip_offset, pos, quat)
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> torch.Tensor:
        """
        The center finger pose is the average of the left and right finger poses.
        """
        left_finger_pose = self.left_finger_pose
        right_finger_pose = self.right_finger_pose
        center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
        center_finger_quat = left_finger_pose[:, 3:7]
        return torch.cat([center_finger_pos, center_finger_quat], dim=-1)
