import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerLockedDoorOpenEnvV2(SawyerXYZEnv):
    def __init__(self, render_mode=None, camera_name=None, camera_id=None):
        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.15)
        obj_high = (0.1, 0.85, 0.15)
        goal_low = (0.0, 0.64, 0.2100)
        goal_high = (0.2, 0.7, 0.2111)

        open_obj_low = (0.0, 0.85, 0.15)
        open_obj_high = (0.1, 0.95, 0.15)
        open_goal_low = (-0.3, 0.4, 0.1499)
        open_goal_high = (-0.2, 0.5, 0.1501)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config = {
            "obj_init_pos": np.array([0, 0.85, 0.15]),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.85, 0.1])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._lock_length = 0.1

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.open_init_config = {
            "obj_init_angle": np.array([0.3]),
            "obj_init_pos": np.array([0.1, 0.95, 0.15]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }
        self.open_goal = np.array([-0.2, 0.7, 0.15])
        self.open_obj_init_pos = self.open_init_config["obj_init_pos"]
        self.open_obj_init_angle = self.open_init_config["obj_init_angle"]
        self.open_hand_init_pos = self.open_init_config["hand_init_pos"]

        self.door_qpos_adr = self.model.joint("doorjoint").qposadr.item()
        self.door_qvel_adr = self.model.joint("doorjoint").dofadr.item()

        self.open_random_reset_space = Box(
            np.array(open_obj_low),
            np.array(open_obj_high),
        )
        self.open_goal_space = Box(np.array(open_goal_low), np.array(open_goal_high))

        self._open_stage_start = None

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_door_lock.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        if self._open_stage_start:
            return self.open_evaluate_state(obs=obs, action=action)
        
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.02)
        info = {
            "success": False,
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(tcp_open > 0),
            "grasp_reward": near_button,
            "in_place_reward": button_pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,

            "unlock_success": success,
        }
        if bool(success):
            self._open_stage_start = True
            self._target_pos = self._open_target_pos.copy()

        return reward, info

    @_assert_task_is_set
    def open_evaluate_state(self, obs, action):
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
        ) = self.open_compute_reward(action, obs)

        success = float(abs(obs[4] - self._open_target_pos[0]) <= 0.08)

        info = {
            "success": success,
            "near_object": reward_ready,
            "grasp_success": reward_grab >= 0.5,
            "grasp_reward": reward_grab,
            "in_place_reward": reward_success,
            "obj_to_target": 0,
            "unscaled_reward": reward,

            "open_success": success,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [
            ("goal_unlock", self._target_pos),
            ("goal_lock", np.array([10.0, 10.0, 10.0])),
        ]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self):
        lock_xpos = self._get_site_pos("lockStartUnlock")
        handle_xpos = self.data.geom("handle").xpos.copy()
        return np.r_[lock_xpos, handle_xpos]

    def _get_quat_objects(self):
        lock_quat = self.data.body("door_link").xquat
        handle_quat = Rotation.from_matrix(
            self.data.geom("handle").xmat.reshape(3, 3)
        ).as_quat()
        return np.r_[lock_quat, handle_quat]

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[self.door_qpos_adr] = pos
        qvel[self.door_qvel_adr] = 0
        self.set_state(qpos, qvel)

    def render_reset(self, obs_info: dict):
        hand_state = obs_info['hand_state'].copy()
        goal_pos = obs_info['goal_pos'].copy()
        lock_pos = obs_info['lock_pos'].copy()
        handle_pos = obs_info['handle_pos'].copy()

        self.render_reset_hand(hand_state=hand_state)

        self.model.body("door").pos = handle_pos
        self._set_obj_xyz(1.5708)

        self.obj_init_pos = lock_pos
        self._target_pos = self.obj_init_pos + np.array([0.1, -0.04, 0.0])

        self.open_objHeight = handle_pos[2]
        self.open_obj_init_pos = self.model.body("door").pos.copy()
        self._open_target_pos = self.open_obj_init_pos + np.array([-0.3, -0.45, 0.0])

        self.model.site("goal").pos = self._open_target_pos

        return self._get_obs()

    def reset_model(self):
        self._reset_hand()
        self.model.body("door").pos = self._get_state_rand_vec()
        self._set_obj_xyz(1.5708)

        self.obj_init_pos = self.data.body("lock_link").xpos
        self._target_pos = self.obj_init_pos + np.array([0.1, -0.04, 0.0])

        self.open_objHeight = self.data.geom("handle").xpos[2]
        self.open_obj_init_pos = self.model.body("door").pos.copy()
        self._open_target_pos = self.open_obj_init_pos + np.array([-0.3, -0.45, 0.0])

        self.model.site("goal").pos = self._open_target_pos
        self.open_maxPullDist = np.linalg.norm(
            self.data.geom("handle").xpos[:-1] - self._open_target_pos[:-1]
        )
        self.target_reward = 1000 * self.open_maxPullDist + 1000 * 2

        return self._get_obs()

    def compute_reward(self, action, obs):
        del action
        gripper = obs[:3]
        lock = obs[4:7]

        # Add offset to track gripper's shoulder, rather than fingers
        offset = np.array([0.0, 0.055, 0.07])

        scale = np.array([0.25, 1.0, 0.5])
        shoulder_to_lock = (gripper + offset - lock) * scale
        shoulder_to_lock_init = (self.init_tcp + offset - self.obj_init_pos) * scale

        # This `ready_to_push` reward should be a *hint* for the agent, not an
        # end in itself. Make sure to devalue it compared to the value of
        # actually unlocking the lock
        ready_to_push = reward_utils.tolerance(
            np.linalg.norm(shoulder_to_lock),
            bounds=(0, 0.02),
            margin=np.linalg.norm(shoulder_to_lock_init),
            sigmoid="long_tail",
        )

        obj_to_target = abs(self._target_pos[0] - lock[0])
        pushed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._lock_length,
            sigmoid="long_tail",
        )

        reward = 2 * ready_to_push + 8 * pushed

        return (
            reward,
            np.linalg.norm(shoulder_to_lock),
            obs[3],
            obj_to_target,
            ready_to_push,
            pushed,
        )

    @staticmethod
    def _reward_grab_effort(actions):
        return (np.clip(actions[3], -1, 1) + 1.0) / 2.0

    @staticmethod
    def _reward_pos(obs, theta):
        hand = obs[:3]
        door = obs[4:7] + np.array([-0.05, 0, 0])

        threshold = 0.12
        # floor is a 3D funnel centered on the door handle
        radius = np.linalg.norm(hand[:2] - door[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = (
            1.0
            if hand[2] >= floor
            else reward_utils.tolerance(
                floor - hand[2],
                bounds=(0.0, 0.01),
                margin=floor / 2.0,
                sigmoid="long_tail",
            )
        )
        # move the hand to a position between the handle and the main door body
        in_place = reward_utils.tolerance(
            np.linalg.norm(hand - door - np.array([0.05, 0.03, -0.01])),
            bounds=(0, threshold / 2.0),
            margin=0.5,
            sigmoid="long_tail",
        )
        ready_to_open = reward_utils.hamacher_product(above_floor, in_place)

        # now actually open the door
        door_angle = -theta
        a = 0.2  # Relative importance of just *trying* to open the door at all
        b = 0.8  # Relative importance of fully opening the door
        opened = a * float(theta < -np.pi / 90.0) + b * reward_utils.tolerance(
            np.pi / 2.0 + np.pi / 6 - door_angle,
            bounds=(0, 0.5),
            margin=np.pi / 3.0,
            sigmoid="long_tail",
        )

        return ready_to_open, opened

    def open_compute_reward(self, actions, obs):
        theta = self.data.joint("doorjoint").qpos

        reward_grab = self._reward_grab_effort(actions)
        reward_steps = self._reward_pos(obs, theta)

        reward = sum(
            (
                2.0 * reward_utils.hamacher_product(reward_steps[0], reward_grab),
                8.0 * reward_steps[1],
            )
        )

        # Override reward on success flag
        reward = reward[0]
        if abs(obs[4] - self._open_target_pos[0]) <= 0.08:
            reward = 10.0

        return (
            reward,
            reward_grab,
            *reward_steps,
        )
