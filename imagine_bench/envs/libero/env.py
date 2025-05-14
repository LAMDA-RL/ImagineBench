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
from stable_baselines3.common.vec_env import SubprocVecEnv

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
        self.state, self.img = self.reset_with_img()
        self.instructions = self.get_instructions(self.env_name)
        self.max_steps = 500
        self.step_count = 0
        self.observation_space = spaces.Box(low=-1, high=1, shape=(44,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.grasped1 = False
        self.grasped2 = False
        self.grasped3 = False
        self.placed1 = False
        self.placed2 = False
        self.placed3 = False
        self.init_state = None
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
        self.grasped1 = False
        self.grasped2 = False
        self.grasped3 = False
        self.placed1 = False
        self.placed2 = False
        self.placed3 = False
        self.init_state = None
        return obs
    
    def reset_with_img(self):
        obs = self.env.reset()
        
        # image = Image.fromarray(obs['agentview_image'])
        # # Save the image as a PNG file
        # image.save(f"output_image_reset7.png")
        # x = aaa
        default_init_states = self.get_default_init_states(self.env_name)
        init_state_id = random.randint(0, len(default_init_states)-1)
        init_state = self.get_init_state(self.env_name, default_init_states, init_state_id)
        # print(self.env.sim.data.qpos.size)
        # print(self.env.sim.data.qvel.size)
        # print(init_state, init_state.shape)
        obs = self.env.set_init_state(init_state)
        # image = Image.fromarray(obs['agentview_image'])
        # # Save the image as a PNG file
        # image.save(f"output_image_reset7.png")
        # x = aaa
        dummy_action = [0.] * 7
        for step in range(5):
            obs, reward, done, info = self.env.step(dummy_action)
        obs, img = self.get_observation(obs)
        self.step_count = 0
        self.grasped1 = False
        self.grasped2 = False
        self.grasped3 = False
        self.placed1 = False
        self.placed2 = False
        self.placed3 = False
        self.init_state = None
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
        done = terminated or truncated
        if terminated:
            info['success'] = True
        else:
            info['success'] = False
        return state, reward, done, info
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
            if env_name in alphabet_soup_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/pick_alphabet_soup_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            elif env_name in cream_cheese_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/pick_cream_cheese_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            elif env_name in salad_dressing_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/pick_salad_dressing_inits.npz"), allow_pickle=True) as data:
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
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/safe_alphabet_soup_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            elif env_name in cream_cheese_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/safe_cream_cheese_inits.npz"), allow_pickle=True) as data:
                    loaded_data_observations = data["observations"]
            elif env_name in salad_dressing_env_name_list:
                with np.load(os.path.join(os.path.dirname(__file__),"./libero_files/safe_salad_dressing_inits.npz"), allow_pickle=True) as data:
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
        if self.init_state is None:
            self.init_state = state
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
            last_robot_pos = last_state[robot_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            reward_dist = last_dist_obj_to_robot - dist_obj_to_robot
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            reward += 20 * reward_dist
            # print(reward, dist_obj_to_robot)
            if dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'pick-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            reward_dist = last_dist_obj_to_robot - dist_obj_to_robot
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            reward += 20 * reward_dist
            if dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'pick-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            reward_dist = last_dist_obj_to_robot - dist_obj_to_robot
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            reward += 20 * reward_dist
            if dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'place-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            reward_dist = last_dist_obj_to_basket - dist_obj_to_basket
            reward += 20 * reward_dist
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.14 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'place-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            reward_dist = last_dist_obj_to_basket - dist_obj_to_basket
            reward += 20 * reward_dist
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.14 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'place-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            reward_dist = last_dist_obj_to_basket - dist_obj_to_basket
            reward += 20 * reward_dist
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.14 and dist_obj_to_basket_z < 0.22:
                reward += 1
                done = True
        elif env_name == 'rep-pick-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            reward_dist = last_dist_obj_to_robot - dist_obj_to_robot
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            reward += 20 * reward_dist
            # print(reward, dist_obj_to_robot)
            if dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'rep-pick-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            reward_dist = last_dist_obj_to_robot - dist_obj_to_robot
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            reward += 20 * reward_dist
            # print(reward, dist_obj_to_robot)
            if dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'rep-pick-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            reward_dist = last_dist_obj_to_robot - dist_obj_to_robot
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            reward += 20 * reward_dist
            # print(reward, dist_obj_to_robot)
            if dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                reward += 1
                done = True
        elif env_name == 'rep-place-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            reward_dist = last_dist_obj_to_basket - dist_obj_to_basket
            reward += 20 * reward_dist
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.14 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'rep-place-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            reward_dist = last_dist_obj_to_basket - dist_obj_to_basket
            reward += 20 * reward_dist
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.14 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'rep-place-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            reward_dist = last_dist_obj_to_basket - dist_obj_to_basket
            reward += 20 * reward_dist
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            if dist_obj_to_basket_xy < 0.14 and dist_obj_to_basket_z < 0.22:
                reward += 1
                done = True
        elif env_name == 'pick-and-place-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = last_dist_obj_to_basket - dist_obj_to_basket
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if dist_obj_to_basket_xy < 0.14 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-and-place-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = last_dist_obj_to_basket - dist_obj_to_basket
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if dist_obj_to_basket_xy < 0.14 and dist_obj_to_basket_z < 0.155:
                reward += 1
                done = True
        elif env_name == 'pick-and-place-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = last_dist_obj_to_basket - dist_obj_to_basket
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if dist_obj_to_basket_xy < 0.14 and dist_obj_to_basket_z < 0.22:
                reward += 1
                done = True
        elif env_name == 'pick-alphabet_soup-and-place-to-cream_cheese':
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[cream_cheese_dim]
            last_basket_pos = last_state[cream_cheese_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = last_dist_obj_to_basket - dist_obj_to_basket
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if dist_obj_to_basket_xy < 0.1 and dist_obj_to_basket_z < 0.22:
                reward += 1
                done = True
        elif env_name == 'pick-alphabet_soup-and-place-to-salad_dressing':
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[salad_dressing_dim]
            last_basket_pos = last_state[salad_dressing_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = last_dist_obj_to_basket - dist_obj_to_basket
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if dist_obj_to_basket_xy < 0.1 and dist_obj_to_basket_z < 0.22:
                reward += 1
                done = True
        elif env_name == 'pick-cream_cheese-and-place-to-alphabet_soup':
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[alphabet_soup_dim]
            last_basket_pos = last_state[alphabet_soup_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = last_dist_obj_to_basket - dist_obj_to_basket
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if dist_obj_to_basket_xy < 0.1 and dist_obj_to_basket_z < 0.22:
                reward += 1
                done = True
        elif env_name == 'pick-cream_cheese-and-place-to-salad_dressing':
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[salad_dressing_dim]
            last_basket_pos = last_state[salad_dressing_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = last_dist_obj_to_basket - dist_obj_to_basket
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if dist_obj_to_basket_xy < 0.1 and dist_obj_to_basket_z < 0.22:
                reward += 1
                done = True
        elif env_name == 'pick-salad_dressing-and-place-to-alphabet_soup':
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[alphabet_soup_dim]
            last_basket_pos = last_state[alphabet_soup_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = last_dist_obj_to_basket - dist_obj_to_basket
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if dist_obj_to_basket_xy < 0.1 and dist_obj_to_basket_z < 0.22:
                reward += 1
                done = True
        elif env_name == 'pick-salad_dressing-and-place-to-cream_cheese':
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[cream_cheese_dim]
            last_basket_pos = last_state[cream_cheese_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            dist_obj_to_basket_xy = np.linalg.norm(obj_pos[0:2] - basket_pos[0:2])
            dist_obj_to_basket_z = np.linalg.norm(obj_pos[2:] - basket_pos[2:])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = last_dist_obj_to_basket - dist_obj_to_basket
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if dist_obj_to_basket_xy < 0.1 and dist_obj_to_basket_z < 0.22:
                reward += 1
                done = True
        elif env_name == 'reach-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            reward_dist = last_dist_obj_to_robot - dist_obj_to_robot
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            reward += 20 * reward_dist
            # print(reward, dist_obj_to_robot)
            if dist_obj_to_robot < 0.05:
                reward += 1
                done = True
        elif env_name == 'reach-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            reward_dist = last_dist_obj_to_robot - dist_obj_to_robot
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            reward += 20 * reward_dist
            # print(reward, dist_obj_to_robot)
            if dist_obj_to_robot < 0.05:
                reward += 1
                done = True
        elif env_name == 'reach-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            reward_dist = last_dist_obj_to_robot - dist_obj_to_robot
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            reward += 20 * reward_dist
            # print(reward, dist_obj_to_robot)
            if dist_obj_to_robot < 0.05:
                reward += 1
                done = True
        elif env_name == 'sequential-pick-and-place-alphabet_soup-and-cream_cheese':

            obj1_pos = state[alphabet_soup_dim]
            last_obj1_pos = last_state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            last_dist_obj1_to_robot = np.linalg.norm(last_obj1_pos - last_robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket = np.linalg.norm(obj1_pos - basket_pos)
            last_dist_obj1_to_basket = np.linalg.norm(last_obj1_pos - last_basket_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            reward_pick1 = last_dist_obj1_to_robot - dist_obj1_to_robot
            reward_place1 = last_dist_obj1_to_basket - dist_obj1_to_basket
            if not self.grasped1 and dist_obj1_to_robot < 0.05 and dist_obj1 > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.placed1 and dist_obj1_to_basket_xy < 0.14 and dist_obj1_to_basket_z < 0.22:
                self.placed1 = True
                reward += 1
            obj2_pos = state[cream_cheese_dim]
            last_obj2_pos = last_state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            last_dist_obj2_to_robot = np.linalg.norm(last_obj2_pos - last_robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket = np.linalg.norm(obj2_pos - basket_pos)
            last_dist_obj2_to_basket = np.linalg.norm(last_obj2_pos - last_basket_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            reward_pick2 = last_dist_obj2_to_robot - dist_obj2_to_robot
            reward_place2 = last_dist_obj2_to_basket - dist_obj2_to_basket
            if not self.grasped2 and dist_obj2_to_robot < 0.05 and dist_obj2 > 0.0001:
                self.grasped2 = True
                reward += 1
            if not self.placed2 and dist_obj2_to_basket_xy < 0.14 and dist_obj2_to_basket_z < 0.22:
                self.placed2 = True
                reward += 1
            if not self.grasped1:
                reward += 20 * reward_pick1
            elif not self.placed1:
                reward += 20 * reward_place1
            elif not self.grasped2:
                reward += 20 * reward_pick2
            elif not self.placed2:
                reward += 20 * reward_place2
            if self.placed1 and self.placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-alphabet_soup-and-salad_dressing':
            obj1_pos = state[alphabet_soup_dim]
            last_obj1_pos = last_state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            last_dist_obj1_to_robot = np.linalg.norm(last_obj1_pos - last_robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket = np.linalg.norm(obj1_pos - basket_pos)
            last_dist_obj1_to_basket = np.linalg.norm(last_obj1_pos - last_basket_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            reward_pick1 = last_dist_obj1_to_robot - dist_obj1_to_robot
            reward_place1 = last_dist_obj1_to_basket - dist_obj1_to_basket
            if not self.grasped1 and dist_obj1_to_robot < 0.05 and dist_obj1 > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.placed1 and dist_obj1_to_basket_xy < 0.14 and dist_obj1_to_basket_z < 0.22:
                self.placed1 = True
                reward += 1
            obj2_pos = state[salad_dressing_dim]
            last_obj2_pos = last_state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            last_dist_obj2_to_robot = np.linalg.norm(last_obj2_pos - last_robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket = np.linalg.norm(obj2_pos - basket_pos)
            last_dist_obj2_to_basket = np.linalg.norm(last_obj2_pos - last_basket_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            reward_pick2 = last_dist_obj2_to_robot - dist_obj2_to_robot
            reward_place2 = last_dist_obj2_to_basket - dist_obj2_to_basket
            if not self.grasped2 and dist_obj2_to_robot < 0.05 and dist_obj2 > 0.0001:
                self.grasped2 = True
                reward += 1
            if not self.placed2 and dist_obj2_to_basket_xy < 0.14 and dist_obj2_to_basket_z < 0.22:
                self.placed2 = True
                reward += 1
            if not self.grasped1:
                reward += 20 * reward_pick1
            elif not self.placed1:
                reward += 20 * reward_place1
            elif not self.grasped2:
                reward += 20 * reward_pick2
            elif not self.placed2:
                reward += 20 * reward_place2
            if self.placed1 and self.placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-cream_cheese-and-alphabet_soup':
            obj1_pos = state[cream_cheese_dim]
            last_obj1_pos = last_state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            last_dist_obj1_to_robot = np.linalg.norm(last_obj1_pos - last_robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket = np.linalg.norm(obj1_pos - basket_pos)
            last_dist_obj1_to_basket = np.linalg.norm(last_obj1_pos - last_basket_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            reward_pick1 = last_dist_obj1_to_robot - dist_obj1_to_robot
            reward_place1 = last_dist_obj1_to_basket - dist_obj1_to_basket
            if not self.grasped1 and dist_obj1_to_robot < 0.05 and dist_obj1 > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.placed1 and dist_obj1_to_basket_xy < 0.14 and dist_obj1_to_basket_z < 0.22:
                self.placed1 = True
                reward += 1
            obj2_pos = state[alphabet_soup_dim]
            last_obj2_pos = last_state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            last_dist_obj2_to_robot = np.linalg.norm(last_obj2_pos - last_robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket = np.linalg.norm(obj2_pos - basket_pos)
            last_dist_obj2_to_basket = np.linalg.norm(last_obj2_pos - last_basket_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            reward_pick2 = last_dist_obj2_to_robot - dist_obj2_to_robot
            reward_place2 = last_dist_obj2_to_basket - dist_obj2_to_basket
            if not self.grasped2 and dist_obj2_to_robot < 0.05 and dist_obj2 > 0.0001:
                self.grasped2 = True
                reward += 1
            if not self.placed2 and dist_obj2_to_basket_xy < 0.14 and dist_obj2_to_basket_z < 0.22:
                self.placed2 = True
                reward += 1
            if not self.grasped1:
                reward += 20 * reward_pick1
            elif not self.placed1:
                reward += 20 * reward_place1
            elif not self.grasped2:
                reward += 20 * reward_pick2
            elif not self.placed2:
                reward += 20 * reward_place2
            if self.placed1 and self.placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-cream_cheese-and-salad_dressing':
            obj1_pos = state[cream_cheese_dim]
            last_obj1_pos = last_state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            last_dist_obj1_to_robot = np.linalg.norm(last_obj1_pos - last_robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket = np.linalg.norm(obj1_pos - basket_pos)
            last_dist_obj1_to_basket = np.linalg.norm(last_obj1_pos - last_basket_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            reward_pick1 = last_dist_obj1_to_robot - dist_obj1_to_robot
            reward_place1 = last_dist_obj1_to_basket - dist_obj1_to_basket
            if not self.grasped1 and dist_obj1_to_robot < 0.05 and dist_obj1 > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.placed1 and dist_obj1_to_basket_xy < 0.14 and dist_obj1_to_basket_z < 0.22:
                self.placed1 = True
                reward += 1
            obj2_pos = state[salad_dressing_dim]
            last_obj2_pos = last_state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            last_dist_obj2_to_robot = np.linalg.norm(last_obj2_pos - last_robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket = np.linalg.norm(obj2_pos - basket_pos)
            last_dist_obj2_to_basket = np.linalg.norm(last_obj2_pos - last_basket_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            reward_pick2 = last_dist_obj2_to_robot - dist_obj2_to_robot
            reward_place2 = last_dist_obj2_to_basket - dist_obj2_to_basket
            if not self.grasped2 and dist_obj2_to_robot < 0.05 and dist_obj2 > 0.0001:
                self.grasped2 = True
                reward += 1
            if not self.placed2 and dist_obj2_to_basket_xy < 0.14 and dist_obj2_to_basket_z < 0.22:
                self.placed2 = True
                reward += 1
            if not self.grasped1:
                reward += 20 * reward_pick1
            elif not self.placed1:
                reward += 20 * reward_place1
            elif not self.grasped2:
                reward += 20 * reward_pick2
            elif not self.placed2:
                reward += 20 * reward_place2
            if self.placed1 and self.placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-salad_dressing-and-alphabet_soup':
            obj1_pos = state[salad_dressing_dim]
            last_obj1_pos = last_state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            last_dist_obj1_to_robot = np.linalg.norm(last_obj1_pos - last_robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket = np.linalg.norm(obj1_pos - basket_pos)
            last_dist_obj1_to_basket = np.linalg.norm(last_obj1_pos - last_basket_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            reward_pick1 = last_dist_obj1_to_robot - dist_obj1_to_robot
            reward_place1 = last_dist_obj1_to_basket - dist_obj1_to_basket
            if not self.grasped1 and dist_obj1_to_robot < 0.05 and dist_obj1 > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.placed1 and dist_obj1_to_basket_xy < 0.14 and dist_obj1_to_basket_z < 0.22:
                self.placed1 = True
                reward += 1
            obj2_pos = state[alphabet_soup_dim]
            last_obj2_pos = last_state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            last_dist_obj2_to_robot = np.linalg.norm(last_obj2_pos - last_robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket = np.linalg.norm(obj2_pos - basket_pos)
            last_dist_obj2_to_basket = np.linalg.norm(last_obj2_pos - last_basket_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            reward_pick2 = last_dist_obj2_to_robot - dist_obj2_to_robot
            reward_place2 = last_dist_obj2_to_basket - dist_obj2_to_basket
            if not self.grasped2 and dist_obj2_to_robot < 0.05 and dist_obj2 > 0.0001:
                self.grasped2 = True
                reward += 1
            if not self.placed2 and dist_obj2_to_basket_xy < 0.14 and dist_obj2_to_basket_z < 0.22:
                self.placed2 = True
                reward += 1
            if not self.grasped1:
                reward += 20 * reward_pick1
            elif not self.placed1:
                reward += 20 * reward_place1
            elif not self.grasped2:
                reward += 20 * reward_pick2
            elif not self.placed2:
                reward += 20 * reward_place2
            if self.placed1 and self.placed2:
                done = True
        elif env_name == 'sequential-pick-and-place-salad_dressing-and-cream_cheese':
            obj1_pos = state[salad_dressing_dim]
            last_obj1_pos = last_state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            last_dist_obj1_to_robot = np.linalg.norm(last_obj1_pos - last_robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket = np.linalg.norm(obj1_pos - basket_pos)
            last_dist_obj1_to_basket = np.linalg.norm(last_obj1_pos - last_basket_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            reward_pick1 = last_dist_obj1_to_robot - dist_obj1_to_robot
            reward_place1 = last_dist_obj1_to_basket - dist_obj1_to_basket
            if not self.grasped1 and dist_obj1_to_robot < 0.05 and dist_obj1 > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.placed1 and dist_obj1_to_basket_xy < 0.14 and dist_obj1_to_basket_z < 0.22:
                self.placed1 = True
                reward += 1
            obj2_pos = state[cream_cheese_dim]
            last_obj2_pos = last_state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            last_dist_obj2_to_robot = np.linalg.norm(last_obj2_pos - last_robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket = np.linalg.norm(obj2_pos - basket_pos)
            last_dist_obj2_to_basket = np.linalg.norm(last_obj2_pos - last_basket_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            reward_pick2 = last_dist_obj2_to_robot - dist_obj2_to_robot
            reward_place2 = last_dist_obj2_to_basket - dist_obj2_to_basket
            if not self.grasped2 and dist_obj2_to_robot < 0.05 and dist_obj2 > 0.0001:
                self.grasped2 = True
                reward += 1
            if not self.placed2 and dist_obj2_to_basket_xy < 0.14 and dist_obj2_to_basket_z < 0.22:
                self.placed2 = True
                reward += 1
            if not self.grasped1:
                reward += 20 * reward_pick1
            elif not self.placed1:
                reward += 20 * reward_place1
            elif not self.grasped2:
                reward += 20 * reward_pick2
            elif not self.placed2:
                reward += 20 * reward_place2
            if self.placed1 and self.placed2:
                done = True
        elif env_name == 'pick-and-place-aside-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_init = np.linalg.norm(obj_pos - self.init_state[alphabet_soup_dim])
            last_dist_obj_to_init = np.linalg.norm(last_obj_pos - self.init_state[alphabet_soup_dim])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = - last_dist_obj_to_init + dist_obj_to_init
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if dist_obj_to_init > 0.07:
                reward += 1
                done = True
        elif env_name == 'pick-and-place-aside-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_init = np.linalg.norm(obj_pos - self.init_state[cream_cheese_dim])
            last_dist_obj_to_init = np.linalg.norm(last_obj_pos - self.init_state[cream_cheese_dim])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = - last_dist_obj_to_init + dist_obj_to_init
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if dist_obj_to_init > 0.07:
                reward += 1
                done = True
            #print(dist_obj_to_init)
        elif env_name == 'pick-and-place-aside-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_init = np.linalg.norm(obj_pos - self.init_state[salad_dressing_dim])
            last_dist_obj_to_init = np.linalg.norm(last_obj_pos - self.init_state[salad_dressing_dim])
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = - last_dist_obj_to_init + dist_obj_to_init
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if dist_obj_to_init > 0.07:
                reward += 1
                done = True
        elif env_name == 'sequential-pick-and-place-all':
            obj1_pos = state[alphabet_soup_dim]
            last_obj1_pos = last_state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj1_to_robot = np.linalg.norm(obj1_pos - robot_pos)
            last_dist_obj1_to_robot = np.linalg.norm(last_obj1_pos - last_robot_pos)
            dist_obj1 = np.linalg.norm(obj1_pos - last_obj1_pos)
            dist_obj1_to_basket = np.linalg.norm(obj1_pos - basket_pos)
            last_dist_obj1_to_basket = np.linalg.norm(last_obj1_pos - last_basket_pos)
            dist_obj1_to_basket_xy = np.linalg.norm(obj1_pos[0:2] - basket_pos[0:2])
            dist_obj1_to_basket_z = np.linalg.norm(obj1_pos[2:] - basket_pos[2:])
            reward_pick1 = last_dist_obj1_to_robot - dist_obj1_to_robot
            reward_place1 = last_dist_obj1_to_basket - dist_obj1_to_basket
            if not self.grasped1 and dist_obj1_to_robot < 0.05 and dist_obj1 > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.placed1 and dist_obj1_to_basket_xy < 0.14 and dist_obj1_to_basket_z < 0.22:
                self.placed1 = True
                reward += 1
            obj2_pos = state[cream_cheese_dim]
            last_obj2_pos = last_state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj2_to_robot = np.linalg.norm(obj2_pos - robot_pos)
            last_dist_obj2_to_robot = np.linalg.norm(last_obj2_pos - last_robot_pos)
            dist_obj2 = np.linalg.norm(obj2_pos - last_obj2_pos)
            dist_obj2_to_basket = np.linalg.norm(obj2_pos - basket_pos)
            last_dist_obj2_to_basket = np.linalg.norm(last_obj2_pos - last_basket_pos)
            dist_obj2_to_basket_xy = np.linalg.norm(obj2_pos[0:2] - basket_pos[0:2])
            dist_obj2_to_basket_z = np.linalg.norm(obj2_pos[2:] - basket_pos[2:])
            reward_pick2 = last_dist_obj2_to_robot - dist_obj2_to_robot
            reward_place2 = last_dist_obj2_to_basket - dist_obj2_to_basket
            if not self.grasped2 and dist_obj2_to_robot < 0.05 and dist_obj2 > 0.0001:
                self.grasped2 = True
                reward += 1
            if not self.placed2 and dist_obj2_to_basket_xy < 0.14 and dist_obj2_to_basket_z < 0.22:
                self.placed2 = True
                reward += 1
            obj3_pos = state[cream_cheese_dim]
            last_obj3_pos = last_state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj3_to_robot = np.linalg.norm(obj3_pos - robot_pos)
            last_dist_obj3_to_robot = np.linalg.norm(last_obj3_pos - last_robot_pos)
            dist_obj3 = np.linalg.norm(obj3_pos - last_obj3_pos)
            dist_obj3_to_basket = np.linalg.norm(obj3_pos - basket_pos)
            last_dist_obj3_to_basket = np.linalg.norm(last_obj3_pos - last_basket_pos)
            dist_obj3_to_basket_xy = np.linalg.norm(obj3_pos[0:2] - basket_pos[0:2])
            dist_obj3_to_basket_z = np.linalg.norm(obj3_pos[2:] - basket_pos[2:])
            reward_pick3 = last_dist_obj3_to_robot - dist_obj3_to_robot
            reward_place3 = last_dist_obj3_to_basket - dist_obj3_to_basket
            if not self.grasped3 and dist_obj3_to_robot < 0.05 and dist_obj3 > 0.0001:
                self.grasped3 = True
                reward += 1
            if not self.placed3 and dist_obj3_to_basket_xy < 0.14 and dist_obj3_to_basket_z < 0.22:
                self.placed3 = True
                reward += 1
            if not self.grasped1:
                reward += 20 * reward_pick1
            elif not self.placed1:
                reward += 20 * reward_place1
            elif not self.grasped2:
                reward += 20 * reward_pick2
            elif not self.placed2:
                reward += 20 * reward_place2
            elif not self.grasped3:
                reward += 20 * reward_pick3
            elif not self.placed3:
                reward += 20 * reward_place3
            if self.placed1 and self.placed2 and self.placed3:
                done = True
        elif env_name == 'pick-out-of-alphabet_soup':
            obj_pos = state[alphabet_soup_dim]
            last_obj_pos = last_state[alphabet_soup_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = dist_obj_to_basket - last_dist_obj_to_basket
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if dist_obj_to_basket > 0.45:
                reward += 1
                done = True
        elif env_name == 'pick-out-of-cream_cheese':
            obj_pos = state[cream_cheese_dim]
            last_obj_pos = last_state[cream_cheese_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = dist_obj_to_basket - last_dist_obj_to_basket
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if dist_obj_to_basket > 0.35:
                reward += 1
                done = True
        elif env_name == 'pick-out-of-salad_dressing':
            obj_pos = state[salad_dressing_dim]
            last_obj_pos = last_state[salad_dressing_dim]
            robot_pos = state[robot_dim]
            last_robot_pos = last_state[robot_dim]
            basket_pos = state[basket_dim]
            last_basket_pos = last_state[basket_dim]
            dist_obj_to_robot = np.linalg.norm(obj_pos - robot_pos)
            last_dist_obj_to_robot = np.linalg.norm(last_obj_pos - last_robot_pos)
            dist_obj = np.linalg.norm(obj_pos - last_obj_pos)
            dist_obj_to_basket = np.linalg.norm(obj_pos - basket_pos)
            last_dist_obj_to_basket = np.linalg.norm(last_obj_pos - last_basket_pos)
            reward_pick = last_dist_obj_to_robot - dist_obj_to_robot
            reward_place = dist_obj_to_basket - last_dist_obj_to_basket
            if not self.grasped1 and dist_obj_to_robot < 0.05 and dist_obj > 0.0001:
                self.grasped1 = True
                reward += 1
            if not self.grasped1:
                reward = reward + 20 * reward_pick
            else:
                reward = reward + 20 * reward_place
            if dist_obj_to_basket > 0.4:
                reward += 1
                done = True
            # print(dist_obj_to_basket)
        else:
            raise ValueError(f"can't find {env_name} instructions")
        return reward, done

class RIMAROEnv:
    action_space: spaces.Box = None
    
    def __init__(self, **kwargs):
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
        
    def init_dataset(self):
        raise NotImplementedError

    def get_dataset(self, **kwargs):
        raise NotImplementedError
    
    def get_instruction(self):
        raise NotImplementedError

class LiberoEnv(RIMAROEnv):
    def __init__(self, **kwargs):
        # self.dataset_url_dict = kwargs['dataset_url_dict']

        self.level = kwargs['level']
        true_level = level2true_level[self.level]
        if true_level == 'baseline':
            self.env_name_list = baseline_env_name_list.copy()
        elif true_level == 'rephrase_level':
            self.env_name_list = rephrase_level_env_name_list.copy()
        elif true_level == 'easy_level':
            self.env_name_list = easy_level_env_name_list.copy()
        elif true_level == 'hard_level':
            self.env_name_list = hard_level_env_name_list.copy()
        else:
            raise NotImplementedError
        # 一个level对应多个env
        self.env_list = []
        for env_name in self.env_name_list:
            eval_env = VectorLibero(env_name)
            self.env_list.append(eval_env)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        self.ptr = None
        self.path_dict = {}
        self.path_dict['real'] = 'data/libero/baseline.npy'
        self.path_dict['rephrase'] = 'data/libero/rephrase.npy'
        self.path_dict['easy'] = 'data/libero/easy.npy'
        self.path_dict['hard'] = 'data/libero/hard.npy'

    def render(self, **kwargs):
        curr_env = self.env_list[self.ptr]
        return curr_env.render(**kwargs)

    def reset(self, **kwargs):
        if self.ptr is None:
            self.ptr = 0
        else:
            self.ptr = (self.ptr + 1) % len(self.env_list)
        curr_env = self.env_list[self.ptr]
        obs = curr_env.reset(**kwargs)
        return obs
    
    def step(self, action):
        curr_env = self.env_list[self.ptr]
        return curr_env.step(action)

    def get_dataset(self, level='rephrase'):
        self.level = level

        # if 'real' not in self.path_dict.keys():
        #     self.path_dict['real'] = download_dataset_from_url(self.dataset_url_dict['real'])
        real_dataset_path = self.path_dict['real']
        np_data = np.load(real_dataset_path, allow_pickle=True).item()
        real_dataset = {
                'masks': np_data['masks'][:],
                'observations': np_data['observations'][:],
                'actions': np_data['actions'][:],
                'rewards': np_data['rewards'][:],
            }
        
        # if self.level not in self.path_dict.keys():
        #     self.path_dict[self.level] = download_dataset_from_url(self.dataset_url_dict[self.level])
        imaginary_level_dataset_path = self.path_dict[self.level]
        np_data = np.load(imaginary_level_dataset_path, allow_pickle=True).item()
        imaginary_level_dataset = {
            'masks': np_data['masks'][:],
            'observations': np_data['observations'][:],
            'actions': np_data['actions'][:],
            'rewards': np_data['rewards'][:],
        }

        return real_dataset, imaginary_level_dataset
    
if __name__ == '__main__':
    # env = VectorLibero("pick-and-place-aside-alphabet_soup")
    # env.reset()
    # for i in range(100):
    #     action = np.random.rand(7)
    #     obs, reward, done, info = env.step(action)
    #     print(obs, reward, done, info)
    #     if done:
    #         break
    env = LiberoEnv(level='real')
    # env.get_dataset(level='real')
    env.reset()
    for i in range(100):
        action = np.random.rand(7)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            break
