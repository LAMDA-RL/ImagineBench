import mujoco
import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerMakeCoffeeEnvV2(SawyerXYZEnv):
    def __init__(self, render_mode=None, camera_name=None, camera_id=None):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.55, -0.001)
        obj_high = (0.1, 0.65, +0.001)
        goal_low = (-0.05, 0.7, -0.001)
        goal_high = (0.05, 0.75, +0.001)

        self.button_max_dist = 0.03
        button_obj_low = (-0.1, 0.8, -0.001)
        button_obj_high = (0.1, 0.9, +0.001)
        button_goal_low = obj_low + np.array([-0.001, -0.22 + self.button_max_dist, 0.299])
        button_goal_high = obj_high + np.array([+0.001, -0.22 + self.button_max_dist, 0.301])

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.6, 0.0]),
            "hand_init_pos": np.array([0.0, 0.4, 0.2]),
        }
        self.goal = np.array([0.0, 0.75, 0])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.button_init_config = {
            "obj_init_pos": np.array([0, 0.9, 0.28]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.4, 0.2]),
        }
        self.button_goal = np.array([0, 0.78, 0.33])
        self.button_obj_init_pos = self.button_init_config["obj_init_pos"]
        self.button_obj_init_angle = self.button_init_config["obj_init_angle"]
        self.button_hand_init_pos = self.button_init_config["hand_init_pos"]
        self.button__random_reset_space = Box(
            np.array(button_obj_low),
            np.array(button_obj_high),
        )
        self.button_goal_space = Box(np.array(button_goal_low), np.array(button_goal_high))
        self._button_target_pos = None
        self._button_stage_start = None

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_coffee.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        if self._button_stage_start:
            return self.button_evaluate_state(obs=obs, action=action)
        
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place,
        ) = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_object and (tcp_open > 0))

        info = {
            "success": False,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,

            "push_success": success,
        }
        if bool(success):
            self._button_stage_start = True
            self._target_pos = self._button_target_pos.copy()

        return reward, info

    @_assert_task_is_set
    def button_evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
        ) = self.button_compute_reward(action, obs)

        success = float(obj_to_target <= 0.02)
        info = {
            "success": success,
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(tcp_open > 0),
            "grasp_reward": near_button,
            "in_place_reward": button_pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,

            "button_success": success,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [("coffee_goal", self._target_pos)]
    
    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self):
        coffee_xpos = self.get_body_com("obj")
        button_xpos = self._get_site_pos("buttonStart")
        return np.r_[coffee_xpos, button_xpos]

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("mug").xmat.reshape(3, 3)
        button_quat = np.array([1.0, 0.0, 0.0, 0.0])
        return np.r_[Rotation.from_matrix(geom_xmat).as_quat(), button_quat]

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        qpos[0:3] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def render_reset(self, obs_info: dict):
        hand_state = obs_info['hand_state'].copy()
        pos_mug_goal = obs_info['goal_pos'].copy()
        pos_mug_init = obs_info['coffee_pos'].copy()

        self.render_reset_hand(hand_state=hand_state)

        self._set_obj_xyz(pos_mug_init)
        self.obj_init_pos = pos_mug_init

        pos_button = obs_info['coffee_button_pos'].copy()
        pos_machine = pos_mug_goal

        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_machine")
        ] = pos_machine

        self._target_pos = pos_mug_goal
        self._button_target_pos = pos_button + np.array([0.0, self.button_max_dist, 0.0])

        return self._get_obs()

    def reset_model(self):
        self._reset_hand()

        pos_mug_init, pos_mug_goal = np.split(self._get_state_rand_vec(), 2)
        while np.linalg.norm(pos_mug_init[:2] - pos_mug_goal[:2]) < 0.15:
            pos_mug_init, pos_mug_goal = np.split(self._get_state_rand_vec(), 2)

        self._set_obj_xyz(pos_mug_init)
        self.obj_init_pos = pos_mug_init

        pos_machine = pos_mug_goal + np.array([0.0, 0.22, 0.0])

        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_machine")
        ] = pos_machine

        self._target_pos = pos_mug_goal

        pos_button = pos_machine + np.array([0.0, -0.22, 0.3])
        self._button_target_pos = pos_button + np.array([0.0, self.button_max_dist, 0.0])
        self._button_stage_start = False

        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        target = self._target_pos.copy()

        # Emphasize X and Y errors
        scale = np.array([2.0, 2.0, 1.0])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, 0.05),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )
        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.04,
            obj_radius=0.02,
            pad_success_thresh=0.05,
            xz_thresh=0.05,
            desired_gripper_effort=0.7,
            medium_density=True,
        )

        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if tcp_to_obj < 0.04 and tcp_opened > 0:
            reward += 1.0 + 5.0 * in_place
        if target_to_obj < 0.05:
            reward = 10.0
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            np.linalg.norm(obj - target),  # recompute to avoid `scale` above
            object_grasped,
            in_place,
        )

    def button_compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._target_pos[1] - obj[1])

        tcp_closed = max(obs[3], 0.0)
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid="long_tail",
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self.button_max_dist,
            sigmoid="long_tail",
        )

        reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed

        return (reward, tcp_to_obj, obs[3], obj_to_target, near_button, button_pressed)