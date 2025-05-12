import gym
from gym import spaces
import numpy as np
import sys
import os
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark
import random
import matplotlib.pyplot as plt
from gymnasium import Env, Wrapper
from gymnasium.core import Any, WrapperObsType, WrapperActType

#easy level


level2true_level = {
    'real': 'baseline',
    'rephrase': 'rephrase_level',
    'easy': 'easy_level',
    'hard': 'hard_level',
}

baseline_env_name_list = [
    'pick-alphabet_soup',
    'pick-cream_cheese', 
    'pick-salad_dressing', 
    'place-alphabet_soup',
    'place-cream_cheese',
    'place-salad_dressing',
]

rephrase_level_env_name_list = [
    'rep-pick-alphabet_soup',
    'rep-pick-cream_cheese', 
    'rep-pick-salad_dressing', 
    'rep-place-alphabet_soup',
    'rep-place-cream_cheese',
    'rep-place-salad_dressing',
]

easy_level_env_name_list = [
    'pick-and-place-alphabet_soup',
    'pick-and-place-cream_cheese',
    'pick-and-place-salad_dressing',
    'pick-alphabet_soup-and-place-to-cream_cheese',
    'pick-alphabet_soup-and-place-to-salad_dressing',
    'pick-cream_cheese-and-place-to-alphabet_soup',
    'pick-cream_cheese-and-place-to-salad_dressing',
    'pick-salad_dressing-and-place-to-alphabet_soup',
    'pick-salad_dressing-and-place-to-cream_cheese',
    'reach-alphabet_soup',
    'reach-cream_cheese',
    'reach-salad_dressing',
]

hard_level_env_name_list = [
    'sequential-pick-and-place-alphabet_soup-and-cream_cheese',
    'sequential-pick-and-place-alphabet_soup-and-salad_dressing',
    'sequential-pick-and-place-cream_cheese-and-alphabet_soup',
    'sequential-pick-and-place-cream_cheese-and-salad_dressing',
    'sequential-pick-and-place-salad_dressing-and-alphabet_soup',
    'sequential-pick-and-place-salad_dressing-and-cream_cheese',
    'pick-and-place-aside-alphabet_soup',
    'pick-and-place-aside-cream_cheese',
    'pick-and-place-aside-salad_dressing',
    'sequential-pick-and-place-all',
    'pick-out-of-alphabet_soup',
    'pick-out-of-cream_cheese',
    'pick-out-of-salad_dressing',
]

alphabet_soup_env_name_list = [
    'pick-alphabet_soup',
    'place-alphabet_soup',
    'rep-pick-alphabet_soup',
    'rep-place-alphabet_soup',
    'pick-and-place-alphabet_soup',
    'pick-alphabet_soup-and-place-to-cream_cheese',
    'pick-alphabet_soup-and-place-to-salad_dressing',
    'reach-alphabet_soup',
    'sequential-pick-and-place-alphabet_soup-and-cream_cheese',
    'sequential-pick-and-place-alphabet_soup-and-salad_dressing',
    'pick-and-place-aside-alphabet_soup',
    'sequential-pick-and-place-all',
    'pick-out-of-alphabet_soup',
]

cream_cheese_env_name_list = [
    'pick-cream_cheese',
    'place-cream_cheese',
    'rep-pick-cream_cheese',
    'rep-place-cream_cheese',
    'pick-and-place-cream_cheese',
    'pick-cream_cheese-and-place-to-alphabet_soup',
    'pick-cream_cheese-and-place-to-salad_dressing',
    'reach-cream_cheese',
    'sequential-pick-and-place-cream_cheese-and-alphabet_soup',
    'sequential-pick-and-place-cream_cheese-and-salad_dressing',
    'pick-and-place-aside-cream_cheese',
    'pick-out-of-cream_cheese',
]

salad_dressing_env_name_list = [
    'pick-salad_dressing',
    'place-salad_dressing',
    'rep-pick-salad_dressing',
    'rep-place-salad_dressing',
    'pick-and-place-salad_dressing',
    'pick-salad_dressing-and-place-to-alphabet_soup',
    'pick-salad_dressing-and-place-to-cream_cheese',
    'reach-salad_dressing',
    'sequential-pick-and-place-salad_dressing-and-alphabet_soup',
    'sequential-pick-and-place-salad_dressing-and-cream_cheese',
    'pick-and-place-aside-salad_dressing',
    'pick-out-of-salad_dressing',
]

pick_env_name_list = [
    'pick-alphabet_soup',
    'pick-cream_cheese', 
    'pick-salad_dressing',
    'rep-pick-alphabet_soup',
    'rep-pick-cream_cheese', 
    'rep-pick-salad_dressing',
    'pick-and-place-alphabet_soup',
    'pick-and-place-cream_cheese',
    'pick-and-place-salad_dressing',
    'pick-alphabet_soup-and-place-to-cream_cheese',
    'pick-alphabet_soup-and-place-to-salad_dressing',
    'pick-cream_cheese-and-place-to-alphabet_soup',
    'pick-cream_cheese-and-place-to-salad_dressing',
    'pick-salad_dressing-and-place-to-alphabet_soup',
    'pick-salad_dressing-and-place-to-cream_cheese',
    'reach-alphabet_soup',
    'reach-cream_cheese',
    'reach-salad_dressing',
    'sequential-pick-and-place-alphabet_soup-and-cream_cheese',
    'sequential-pick-and-place-alphabet_soup-and-salad_dressing',
    'sequential-pick-and-place-cream_cheese-and-alphabet_soup',
    'sequential-pick-and-place-cream_cheese-and-salad_dressing',
    'sequential-pick-and-place-salad_dressing-and-alphabet_soup',
    'sequential-pick-and-place-salad_dressing-and-cream_cheese',
    'pick-and-place-aside-alphabet_soup',
    'pick-and-place-aside-cream_cheese',
    'pick-and-place-aside-salad_dressing',
    'sequential-pick-and-place-all',
]

place_env_name_list = [
    'place-alphabet_soup',
    'place-cream_cheese',
    'place-salad_dressing',
    'rep-place-alphabet_soup',
    'rep-place-cream_cheese',
    'rep-place-salad_dressing',
]

safe_env_name_list = [
    'pick-out-of-alphabet_soup',
    'pick-out-of-cream_cheese',
    'pick-out-of-salad_dressing',
]




class VectorLibero(gym.Env):
    def __init__(self, env_name):
        super(VectorLibero, self).__init__()
        self.env_name = env_name
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, etc.
        self.task_bddl_file = self.get_bbdl_files(env_name)

        env_args = {
            "bddl_file_name": self.task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128
        }
        self.env = OffScreenRenderEnv(**env_args)
        self.env.seed(0)
        self.state, self.img = self.reset()
        self.instructions = self.get_instructions(self.env_name)
        self.max_steps = 500
        self.step_count = 0

    def reset(self):
        self.env.reset()
        default_init_states = self.get_default_init_states(self.env_name)
        init_state_id = random.randint(0, len(default_init_states)-1)
        init_state = self.get_init_state(self.env_name, default_init_states, init_state_id)
        obs = self.env.set_init_state(init_state)
        dummy_action = [0.] * 7
        for step in range(5):
            obs, reward, done, info = self.env.step(dummy_action)
        obs, img = self.get_observation(obs)
        self.step_count = 0
        return obs, img
    
    def get_observation(self, obs):
        img = obs["agentview_image"]
        robot_obs = np.concatenate([obs["robot0_joint_pos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]], axis=0)
        obj0_obs = np.concatenate([obs["alphabet_soup_1_pos"], obs["alphabet_soup_1_quat"]], axis=0)
        obj1_obs = np.concatenate([obs["cream_cheese_1_pos"], obs["cream_cheese_1_quat"]], axis=0)
        obj2_obs = np.concatenate([obs["salad_dressing_1_pos"], obs["salad_dressing_1_quat"]], axis=0)
        basket_obs = np.concatenate([obs["basket_1_pos"], obs["basket_1_quat"]], axis=0)
        state = np.concatenate([robot_obs, obj0_obs, obj1_obs, obj2_obs, basket_obs], axis=0)
        return state, img
    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        self.step_count += 1
        state, img = self.get_observation(obs)
        reward, terminated = self.get_reward_and_done(state, self.state, self.env_name)
        if self.step_count >= self.max_steps:
            truncated = True
        else:
            truncated = False
        # if terminated or truncated:
        #     self.env.close()
        self.state = state
        self.img = obs["agentview_image"]
        return state, reward, terminated, truncated, info
    def render(self):
        return self.img
    
    def get_mujoco_state(self, state):
        mujoco_state = np.zeros(71)
        mujoco_state[1:8] = state[0:7]
        mujoco_state[8:10] = state[14:16]
        obj1_state, obj2_state, obj3_state, basket_state = state[16:23], state[23:30], state[30:37], state[37:]
        mujoco_state[10:10+7] = self.pos_quat_transfer(obj1_state)
        mujoco_state[10+7:10+7+7] = self.pos_quat_transfer(basket_state)
        mujoco_state[10+7+7:10+7+7+7] = self.pos_quat_transfer(obj3_state)
        mujoco_state[10+7+7+7:10+7+7+7+7] = self.pos_quat_transfer(obj2_state)
        return mujoco_state

    def set_render_state(self, state):
        mujoco_state = self.get_mujoco_state(state)
        obs = self.env.set_init_state(mujoco_state)
        self.state = state
        self.img = obs["agentview_image"]
        return obs["agentview_image"]
    def get_instructions(self, env_name):
        if env_name == 'pick-alphabet_soup':
            instructions = self.pick_instructions('alphabet_soup')
        elif env_name == 'pick-cream_cheese':
            instructions = self.pick_instructions('cream_cheese')
        elif env_name == 'pick-salad_dressing':
            instructions = self.pick_instructions('salad_dressing')
        elif env_name == 'place-alphabet_soup':
            instructions = self.place_instructions('alphabet_soup')
        elif env_name == 'place-cream_cheese':
            instructions = self.place_instructions('cream_cheese')
        elif env_name == 'place-salad_dressing':
            instructions = self.place_instructions('salad_dressing')
        elif env_name == 'rep-pick-alphabet_soup':
            instructions = self.unseen_pick_instructions('alphabet_soup')
        elif env_name == 'rep-pick-cream_cheese':
            instructions = self.unseen_pick_instructions('cream_cheese')
        elif env_name == 'rep-pick-salad_dressing':
            instructions = self.unseen_pick_instructions('salad_dressing')
        elif env_name == 'rep-place-alphabet_soup':
            instructions = self.unseen_place_instructions('alphabet_soup')
        elif env_name == 'rep-place-cream_cheese':
            instructions = self.unseen_place_instructions('cream_cheese')
        elif env_name == 'rep-place-salad_dressing':
            instructions = self.unseen_place_instructions('salad_dressing')
        elif env_name == 'pick-and-place-alphabet_soup':
            instructions = self.pick_and_place_instructions('alphabet_soup')
        elif env_name == 'pick-and-place-cream_cheese':
            instructions = self.pick_and_place_instructions('cream_cheese')
        elif env_name == 'pick-and-place-salad_dressing':
            instructions = self.pick_and_place_instructions('salad_dressing')
        elif env_name == 'pick-alphabet_soup-and-place-to-cream_cheese':
            instructions = self.pick_obj1_and_place_obj2_instructions('alphabet_soup', 'cream_cheese')
        elif env_name == 'pick-alphabet_soup-and-place-to-salad_dressing':
            instructions = self.pick_obj1_and_place_obj2_instructions('alphabet_soup', 'salad_dressing')
        elif env_name == 'pick-cream_cheese-and-place-to-alphabet_soup':
            instructions = self.pick_obj1_and_place_obj2_instructions('cream_cheese', 'alphabet_soup')
        elif env_name == 'pick-cream_cheese-and-place-to-salad_dressing':
            instructions = self.pick_obj1_and_place_obj2_instructions('cream_cheese', 'salad_dressing')
        elif env_name == 'pick-salad_dressing-and-place-to-alphabet_soup':
            instructions = self.pick_obj1_and_place_obj2_instructions('salad_dressing', 'alphabet_soup')
        elif env_name == 'pick-salad_dressing-and-place-to-cream_cheese':
            instructions = self.pick_obj1_and_place_obj2_instructions('salad_dressing', 'cream_cheese')
        elif env_name == 'reach-alphabet_soup':
            instructions = self.reach_instructions('alphabet_soup')
        elif env_name == 'reach-cream_cheese':
            instructions = self.reach_instructions('cream_cheese')
        elif env_name == 'reach-salad_dressing':
            instructions = self.reach_instructions('salad_dressing')
        elif env_name == 'sequential-pick-and-place-alphabet_soup-and-cream_cheese':
            instructions = self.sequential_pick_and_place_instructions('alphabet_soup', 'cream_cheese')
        elif env_name == 'sequential-pick-and-place-alphabet_soup-and-salad_dressing':
            instructions = self.sequential_pick_and_place_instructions('alphabet_soup', 'salad_dressing')
        elif env_name == 'sequential-pick-and-place-cream_cheese-and-alphabet_soup':
            instructions = self.sequential_pick_and_place_instructions('cream_cheese', 'alphabet_soup')
        elif env_name == 'sequential-pick-and-place-cream_cheese-and-salad_dressing':
            instructions = self.sequential_pick_and_place_instructions('cream_cheese', 'salad_dressing')
        elif env_name == 'sequential-pick-and-place-salad_dressing-and-alphabet_soup':
            instructions = self.sequential_pick_and_place_instructions('salad_dressing', 'alphabet_soup')
        elif env_name == 'sequential-pick-and-place-salad_dressing-and-cream_cheese':
            instructions = self.sequential_pick_and_place_instructions('salad_dressing', 'cream_cheese')
        elif env_name == 'pick-and-place-aside-alphabet_soup':
            instructions = self.pick_and_place_aside_instructions('alphabet_soup')
        elif env_name == 'pick-and-place-aside-cream_cheese':
            instructions = self.pick_and_place_aside_instructions('cream_cheese')
        elif env_name == 'pick-and-place-aside-salad_dressing':
            instructions = self.pick_and_place_aside_instructions('salad_dressing')
        elif env_name == 'sequential-pick-and-place-all':
            instructions = self.pick_and_place_all_instructions()
        elif env_name == 'pick-out-of-alphabet_soup':
            instructions = self.safe_instructions('alphabet_soup')
        elif env_name == 'pick-out-of-cream_cheese':
            instructions = self.safe_instructions('cream_cheese')
        elif env_name == 'pick-out-of-salad_dressing':
            instructions = self.safe_instructions('salad_dressing')
        else:
            raise ValueError(f"can't find {env_name} instructions")
        return instructions
    
    def get_init_state(self, env_name, default_init_states, init_state_id):
        default_init_state = default_init_states[init_state_id]
        if env_name in pick_env_name_list:
            task_bddl_file = os.path.join(os.path.dirname(__file__),"./libero_files/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl")

            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            }
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)
            env.reset()
            obs = env.set_init_state(default_init_state)
            robot_state = default_init_state[0:10]
            obj1_state = np.concatenate([obs["alphabet_soup_1_pos"], obs["alphabet_soup_1_quat"]], axis=0)
            obj2_state = np.concatenate([obs["cream_cheese_1_pos"], obs["cream_cheese_1_quat"]], axis=0)
            if env_name in cream_cheese_env_name_list:
                obj3_state = np.concatenate([obs["tomato_sauce_1_pos"], obs["tomato_sauce_1_quat"]], axis=0)
            else:
                obj3_state = np.concatenate([obs["salad_dressing_1_pos"], obs["salad_dressing_1_quat"]], axis=0)
            basket_state = np.concatenate([obs["basket_1_pos"], obs["basket_1_quat"]], axis=0)
            init_state = np.zeros(71)
            init_state[0:10] = robot_state
            init_state[10:10+7] = self.pos_quat_transfer(obj1_state)
            init_state[10+7:10+7+7] = self.pos_quat_transfer(basket_state)
            init_state[10+7+7:10+7+7+7] = self.pos_quat_transfer(obj3_state)
            init_state[10+7+7+7:10+7+7+7+7] = self.pos_quat_transfer(obj2_state)
            env.close()
        elif env_name in place_env_name_list:
            if env_name in alphabet_soup_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/place_alphabet_soup_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            elif env_name in cream_cheese_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/place_cream_cheese_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            elif env_name in salad_dressing_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/place_salad_dressing_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            state = loaded_data_observations[random.choice(range(0, 1000))]
            init_state = np.zeros(71)
            init_state[1:8] = state[0:7]
            init_state[8:10] = state[14:16]
            obj1_state, obj2_state, obj3_state, basket_state = state[16:23], state[23:30], state[30:37], state[37:]
            init_state[10:10+7] = self.pos_quat_transfer(obj1_state)
            init_state[10+7:10+7+7] = self.pos_quat_transfer(basket_state)
            init_state[10+7+7:10+7+7+7] = self.pos_quat_transfer(obj3_state)
            init_state[10+7+7+7:10+7+7+7+7] = self.pos_quat_transfer(obj2_state)
        elif env_name in safe_env_name_list:
            if env_name in alphabet_soup_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/place_alphabet_soup_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            elif env_name in cream_cheese_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/place_cream_cheese_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            elif env_name in salad_dressing_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/place_salad_dressing_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            state = loaded_data_observations[random.choice(range(0, 1000))]
            init_state = np.zeros(71)
            init_state[1:8] = state[0:7]
            init_state[8:10] = state[14:16]
            obj1_state, obj2_state, obj3_state, basket_state = state[16:23], state[23:30], state[30:37], state[37:]
            init_state[10:10+7] = self.pos_quat_transfer(obj1_state)
            init_state[10+7:10+7+7] = self.pos_quat_transfer(basket_state)
            init_state[10+7+7:10+7+7+7] = self.pos_quat_transfer(obj3_state)
            init_state[10+7+7+7:10+7+7+7+7] = self.pos_quat_transfer(obj2_state)
        return init_state

    def get_default_init_states(self, env_name):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, etc.
        task_suite = benchmark_dict[task_suite_name]()
        if env_name in alphabet_soup_env_name_list:
            task_id = 0
        elif env_name in cream_cheese_env_name_list:
            task_id = 1
        elif env_name in salad_dressing_env_name_list:
            task_id = 2
        init_states = task_suite.get_task_init_states(task_id)
        # init_state_id = random.randint(0, len(init_states)-1)
        # init_state = init_states[init_state_id]
        return init_states

    def pos_quat_transfer(self, state):
        mujoco_state = np.zeros(7)
        mujoco_state[0:3] = state[0:3]
        mujoco_state[3] = state[-1]
        mujoco_state[4:7] = state[3:6]
        return mujoco_state
    
    def get_bbdl_files(self, env_name):
        if env_name in alphabet_soup_env_name_list:
            task_bddl_file = os.path.join(os.path.dirname(__file__),"./libero_files/init.bddl")
        elif env_name in cream_cheese_env_name_list:
            task_bddl_file = os.path.join(os.path.dirname(__file__),"./libero_files/init.bddl")
        elif env_name in salad_dressing_env_name_list:
            task_bddl_file = os.path.join(os.path.dirname(__file__),"./libero_files/init.bddl")
        else:
            raise ValueError(f"can't find {env_name} bbdl file")
        return task_bddl_file
    
    def pick_and_place_instructions(self, obj):
        instruction = f'Employ the gripper to seize the {obj} and transfer the {obj} to the basket.'
        return [instruction for _ in range(10)]

    def pick_obj1_and_place_obj2_instructions(self, obj1, obj2):
        instruction = f'Employ the gripper to seize the {obj1} and transfer the {obj1} to the {obj2}.'
        return [instruction for _ in range(10)]

    def reach_instructions(self, obj):
        instruction = f'Employ the gripper to get close to the {obj}.'
        return [instruction for _ in range(10)]

    def sequential_pick_and_place_instructions(self, obj1, obj2):
        instruction = f'Employ the gripper to seize the {obj1} and transfer the {obj1} to the basket. Then employ the gripper to seize the {obj2} and transfer the {obj2} to the basket.'
        return [instruction for _ in range(10)]

    def pick_and_place_aside_instructions(self, obj):
        instruction = f'Employ the gripper to seize the {obj} and transfer the {obj} to the other side.'
        return [instruction for _ in range(10)]

    def pick_and_place_all_instructions(self, ):
        instruction = f'Employ the gripper to seize something and transfer it to the basket one by one until the alphabet_soup, cream_cheese and salad_dressing are all in the basket.'
        return [instruction for _ in range(10)]

    def safe_instructions(self, obj):
        instruction = f'The basket is on fire, employ the gripper to seize the {obj} in the basket and transfer the {obj} out of the basket.'
        return [instruction for _ in range(10)]

    def unseen_place_instructions(self, obj):
        instructions = [
        f'Transport the {obj} to the basket.',
        f'Insert the {obj} into the basket.',
        f'Drop off the {obj} in the basket.',
        f'Settle the {obj} into the basket.',
        f'Arrange the {obj} into the basket.',
        f'Guide the {obj} into the basket.',
        f'Park the {obj} in the basket.',
        f'Fit the {obj} inside the basket.',
        f'Dispatch the {obj} to the basket.',
        f'Unload the {obj} into the basket.',
        ]
        return instructions

    def unseen_pick_instructions(self, obj):
        instructions = [
        f'Employ the gripper tool to clasp the {obj}.',
        f'Utilize the gripping mechanism to hold the {obj}.',
        f'Activate the gripper system to seize the {obj}.',
        f'Engage the robotic gripper to grasp the {obj}.',
        f'Deploy the gripping apparatus to capture the {obj}.',
        f'Leverage the gripper mechanism to hold the {obj}.',
        f'Use the robotic gripping system to clutch the {obj}.',
        f'Apply the mechanical gripper to secure the {obj}.',
        f'Operate the gripper device to clasp the {obj}.',
        f'Activate the robotic gripper to grasp the {obj}.',
        ]
        return instructions

    def place_instructions(self, obj):
        instructions = [f'Transfer the {obj} to the basket.',
                        f'Shift the {obj} to the basket.',
                        f'Position the {obj} to the basket.',
                        f'Move the {obj} to the basket.',
                        f'Place the {obj} to the basket.',
                        f'Relocate the {obj} to the basket.',
                        f'Deposit the {obj} to the basket.',
                        f'Put the {obj} in the basket.',
                        f'Drop the {obj} into the basket.',
                        f'Deliver the {obj} to the basket.',
                        f'Set the {obj} in the basket.',
                        f'Lay the {obj} in the basket.',
                        f'Load the {obj} into the basket.',
                        f'Carry the {obj} to the basket.',
                        f'Convey the {obj} to the basket.',
                        f'Shift the {obj} into the basket.',
                        f'Arrange the {obj} in the basket.',
                        f'Displace the {obj} to the basket.',
                        f'Drag the {obj} to the basket.',
                        f'Guide the {obj} to the basket.']
        return instructions

    def pick_instructions(self, obj):
        instructions = [f'Employ the gripper to seize the {obj}.',
            f'Utilize the gripper for grasping the {obj}.',
            f'Employ the gripper mechanism to grasp the {obj}.',
            f'Apply the gripper tool to grasp the {obj}.',
            f'Use the gripper device to seize the {obj}.',
            f'Utilize the gripper apparatus to capture the {obj}.',
            f'Employ the gripper mechanism to clasp the {obj}.'
            f'Engage the gripper to grasp the {obj}.',
            f'Deploy the gripper to seize the {obj}.',
            f'Make use of the gripper to hold the {obj}.',
            f'Operate the gripper to grasp the {obj}.',
            f'Activate the gripper to clutch the {obj}.',
            f'Employ the robotic gripper to capture the {obj}.',
            f'Use the gripping tool to secure the {obj}.',
            f'Utilize the gripper attachment to grasp the {obj}.',
            f'Apply the gripping mechanism to seize the {obj}.',
            f'Leverage the gripper device to clasp the {obj}.',
            f'Engage the gripper mechanism to secure the {obj}.',
            f'Activate the gripping device to hold the {obj}.',
            f'Deploy the gripper tool to clasp the {obj}.',
            f'Use the robotic gripper to grasp the {obj}.']
        return instructions
    
    def get_reward_and_done(self, state, last_state, env_name):
        done = False
        reward = 0
        robot_dim = [7,8,9]
        alphabet_soup_dim = [16,17,18]
        cream_cheese_dim = [23,24,25]
        salad_dressing_dim = [30,31,32]
        basket_dim = [37,38,39]
        if env_name == 'pick-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            if dist_obj_to_robot < 0.025 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'pick-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            if dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'pick-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            if dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'place-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            basket_pos = state[basket_dim]
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'place-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            basket_pos = state[basket_dim]
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'place-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            basket_pos = state[basket_dim]
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'rep-pick-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            if dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'rep-pick-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            if dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'rep-pick-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            if dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'rep-place-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            basket_pos = state[basket_dim]
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'rep-place-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            basket_pos = state[basket_dim]
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'rep-place-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            basket_pos = state[basket_dim]
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-and-place-alphabet_soup':
            grasped = False
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-and-place-cream_cheese':
            grasped = False
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-and-place-salad_dressing':
            grasped = False
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-alphabet_soup-and-place-to-cream_cheese':
            grasped = False
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[cream_cheese_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-alphabet_soup-and-place-to-salad_dressing':
            grasped = False
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[salad_dressing_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-cream_cheese-and-place-to-alphabet_soup':
            grasped = False
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[alphabet_soup_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-cream_cheese-and-place-to-salad_dressing':
            grasped = False
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[salad_dressing_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-salad_dressing-and-place-to-alphabet_soup':
            grasped = False
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[alphabet_soup_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-salad_dressing-and-place-to-cream_cheese':
            grasped = False
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[cream_cheese_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket_xy < 0.05 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'reach-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            if dist_obj_to_robot < 0.03:
                reward += 1
                done = True
        elif env_name == 'reach-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            if dist_obj_to_robot < 0.03:
                reward += 1
                done = True
        elif env_name == 'reach-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            if dist_obj_to_robot < 0.03:
                reward += 1
                done = True
        elif env_name == 'sequential-pick-and-place-alphabet_soup-and-cream_cheese':
            grasped1 = False
            placed1 = False
            obj1_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj1_pos = last_state[alphabet_soup_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            if not grasped1 and dist_obj1_to_robot < 0.03 and dist_obj1 > 0.0001:
                grasped1 = True
                reward += 1
            if not placed1 and dist_obj1_to_basket_xy < 0.05 and dist_obj1_to_basket_z < 0.155:
                placed1 = True
                reward += 1
            grasped2 = False
            placed2 = False
            obj2_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj2_pos = last_state[cream_cheese_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            if not grasped2 and dist_obj2_to_robot < 0.03 and dist_obj2 > 0.0001:
                grasped2 = True
                reward += 1
            if not placed2 and dist_obj2_to_basket_xy < 0.05 and dist_obj2_to_basket_z < 0.155:
                placed2 = True
                reward += 1
            if placed1 and placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-alphabet_soup-and-salad_dressing':
            grasped1 = False
            placed1 = False
            obj1_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj1_pos = last_state[alphabet_soup_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            if not grasped1 and dist_obj1_to_robot < 0.03 and dist_obj1 > 0.0001:
                grasped1 = True
                reward += 1
            if not placed1 and dist_obj1_to_basket_xy < 0.05 and dist_obj1_to_basket_z < 0.155:
                placed1 = True
                reward += 1
            grasped2 = False
            placed2 = False
            obj2_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj2_pos = last_state[salad_dressing_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            if not grasped2 and dist_obj2_to_robot < 0.03 and dist_obj2 > 0.0001:
                grasped2 = True
                reward += 1
            if not placed2 and dist_obj2_to_basket_xy < 0.05 and dist_obj2_to_basket_z < 0.155:
                placed2 = True
                reward += 1
            if placed1 and placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-cream_cheese-and-alphabet_soup':
            grasped1 = False
            placed1 = False
            obj1_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj1_pos = last_state[cream_cheese_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            if not grasped1 and dist_obj1_to_robot < 0.03 and dist_obj1 > 0.0001:
                grasped1 = True
                reward += 1
            if not placed1 and dist_obj1_to_basket_xy < 0.05 and dist_obj1_to_basket_z < 0.155:
                placed1 = True
                reward += 1
            grasped2 = False
            placed2 = False
            obj2_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj2_pos = last_state[alphabet_soup_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            if not grasped2 and dist_obj2_to_robot < 0.03 and dist_obj2 > 0.0001:
                grasped2 = True
                reward += 1
            if not placed2 and dist_obj2_to_basket_xy < 0.05 and dist_obj2_to_basket_z < 0.155:
                placed2 = True
                reward += 1
            if placed1 and placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-cream_cheese-and-salad_dressing':
            grasped1 = False
            placed1 = False
            obj1_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj1_pos = last_state[cream_cheese_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            if not grasped1 and dist_obj1_to_robot < 0.03 and dist_obj1 > 0.0001:
                grasped1 = True
                reward += 1
            if not placed1 and dist_obj1_to_basket_xy < 0.05 and dist_obj1_to_basket_z < 0.155:
                placed1 = True
                reward += 1
            grasped2 = False
            placed2 = False
            obj2_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj2_pos = last_state[salad_dressing_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            if not grasped2 and dist_obj2_to_robot < 0.03 and dist_obj2 > 0.0001:
                grasped2 = True
                reward += 1
            if not placed2 and dist_obj2_to_basket_xy < 0.05 and dist_obj2_to_basket_z < 0.155:
                placed2 = True
                reward += 1
            if placed1 and placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-salad_dressing-and-alphabet_soup':
            grasped1 = False
            placed1 = False
            obj1_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj1_pos = last_state[salad_dressing_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            if not grasped1 and dist_obj1_to_robot < 0.03 and dist_obj1 > 0.0001:
                grasped1 = True
                reward += 1
            if not placed1 and dist_obj1_to_basket_xy < 0.05 and dist_obj1_to_basket_z < 0.155:
                placed1 = True
                reward += 1
            grasped2 = False
            placed2 = False
            obj2_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj2_pos = last_state[alphabet_soup_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            if not grasped2 and dist_obj2_to_robot < 0.03 and dist_obj2 > 0.0001:
                grasped2 = True
                reward += 1
            if not placed2 and dist_obj2_to_basket_xy < 0.05 and dist_obj2_to_basket_z < 0.155:
                placed2 = True
                reward += 1
            if placed1 and placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-salad_dressing-and-cream_cheese':
            grasped1 = False
            placed1 = False
            obj1_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj1_pos = last_state[salad_dressing_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            if not grasped1 and dist_obj1_to_robot < 0.03 and dist_obj1 > 0.0001:
                grasped1 = True
                reward += 1
            if not placed1 and dist_obj1_to_basket_xy < 0.05 and dist_obj1_to_basket_z < 0.155:
                placed1 = True
                reward += 1
            grasped2 = False
            placed2 = False
            obj2_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj2_pos = last_state[cream_cheese_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            if not grasped2 and dist_obj2_to_robot < 0.03 and dist_obj2 > 0.0001:
                grasped2 = True
                reward += 1
            if not placed2 and dist_obj2_to_basket_xy < 0.05 and dist_obj2_to_basket_z < 0.155:
                placed2 = True
                reward += 1
            if placed1 and placed2:
                done = True
        elif env_name == 'pick-and-place-aside-alphabet_soup':
            grasped = False
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_o = np.linalg.norm(obj_pos - np.array([0.0, 0.0, 0.0]))
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_o < 0.03:
                reward += 1
                done = True
        elif env_name == 'pick-and-place-aside-cream_cheese':
            grasped = False
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_o = np.linalg.norm(obj_pos - np.array([0.0, 0.0, 0.0]))
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_o < 0.03:
                reward += 1
                done = True
        elif env_name == 'pick-and-place-aside-salad_dressing':
            grasped = False
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_o = np.linalg.norm(obj_pos - np.array([0.0, 0.0, 0.0]))
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_o < 0.03:
                reward += 1
                done = True
        elif env_name == 'sequential-pick-and-place-all':
            grasped1 = False
            placed1 = False
            obj1_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj1_pos = last_state[alphabet_soup_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            if not grasped1 and dist_obj1_to_robot < 0.03 and dist_obj1 > 0.0001:
                grasped1 = True
                reward += 1
            if not placed1 and dist_obj1_to_basket_xy < 0.05 and dist_obj1_to_basket_z < 0.155:
                placed1 = True
                reward += 1
            grasped2 = False
            placed2 = False
            obj2_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj2_pos = last_state[cream_cheese_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            if not grasped2 and dist_obj2_to_robot < 0.03 and dist_obj2 > 0.0001:
                grasped2 = True
                reward += 1
            if not placed2 and dist_obj2_to_basket_xy < 0.05 and dist_obj2_to_basket_z < 0.155:
                placed2 = True
                reward += 1
            grasped3 = False
            placed3 = False
            obj3_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj3_pos = last_state[salad_dressing_dim]
            dist_obj3_to_robot = np.linalg.norm(obj3_pos - robot_pos)
            dist_obj3 = np.linalg.norm(obj3_pos - last_obj3_pos)
            dist_obj3_to_basket_xy = np.linalg.norm(obj3_pos[0:2] - basket_pos[0:2])
            dist_obj3_to_basket_z = np.linalg.norm(obj3_pos[2:] - basket_pos[2:])
            if not grasped3 and dist_obj3_to_robot < 0.03 and dist_obj3 > 0.0001:
                grasped3 = True
                reward += 1
            if not placed3 and dist_obj3_to_basket_xy < 0.05 and dist_obj3_to_basket_z < 0.155:
                placed3 = True
                reward += 1
            if placed1 and placed2 and placed3:
                done = True
        elif env_name == 'pick-out-of-alphabet_soup':
            grasped = False
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket > 0.3:
                reward += 1
                done = True
        elif env_name == 'pick-out-of-cream_cheese':
            grasped = False
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket > 0.3:
                reward += 1
                done = True
        elif env_name == 'pick-out-of-salad_dressing':
            grasped = False
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            basket_pos = state[basket_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            if not grasped and dist_obj_to_robot < 0.03 and dist_obj > 0.0001:
                grasped = True
                reward += 1
            if dist_obj_to_basket > 0.3:
                reward += 1
                done = True
        else:
            raise ValueError(f"can't find {env_name} instructions")
        return reward, done

if __name__ == "__main__":
    env_name = 'place-salad_dressing'
    env = VectorLibero(env_name)
    env.reset()
    done = False
    while not done:
        dummy_action = [0.] * 7
        state, reward, terminated, truncated, info = env.step(dummy_action)
        print(state.shape)
    # env, img = make_libero_env(env_name)
    # env = VectorLibero(env, env_name)
    # state, img = env.reset()
    # # plt.imshow(img[::-1, :, :])
    # # plt.axis("off")  # Turn off axes for a cleaner image
    # # plt.savefig("test_image2.png", bbox_inches="tight")
    # dummy_action = [0.] * 7
    # state, reward, done, info = env.step(dummy_action)
    # print(state)
    # env.env.close()