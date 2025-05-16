
import numpy as np

from gym import spaces
import gym

level2true_level = {
    'real': 'baseline',
    'rephrase': 'rephrase_level',
    'easy': 'easy_level',
    'hard': 'hard_level',
}

jump_env_name_list = [
    'jump-forward',
    'jump-backward',
    'rep-jump-forward',
    'rep-jump-backward',
]

move_env_name_list = [
    'move-forward',
    'move-backward',
    'rep-move-forward',
    'rep-move-backward'
]

baseline_mujoco_env_name_list = [
    'jump-forward',
    'jump-backward',
    'move-forward',
    'move-backward'
]

move_env_name_list2 = [
    'move-forward-fast',
    'move-backward-fast',
    'move-forward-backward',
    'move-backward-forward',
]

forward_env_name_list = [
    'jump-forward',
    'move-forward',
    'rep-jump-forward',
    'rep-move-forward',
    'move-forward-fast',
]

backward_env_name_list = [
    'jump-backward',
    'move-backward',
    'rep-jump-backward',
    'rep-move-backward',
    'move-backward-fast',
]

forward_backward_env_name_list = [
    'move-forward-backward',
    'move-backward-forward',
]

rephrase_level_mujoco_env_name_list = [
    'rep-jump-forward',
    'rep-jump-backward',
    'rep-move-forward',
    'rep-move-backward'
]

easy_level_mujoco_env_name_list = [
    'move-forward-fast',
    'move-backward-fast',
]

hard_level_mujoco_env_name_list = [
    'move-forward-backward',
    'move-backward-forward',
    'jump-in-place'
]

jump_forward_instructions = ['Jump a step forward.',
                    # 'Jump a step ahead.',
                    # 'Leap a step forward.',
                    # 'Leap a step ahead.',
                    # 'Hop a step forward.',
                    # 'Hop a step ahead.',
                    # 'Spring a step forward.',
                    # 'Spring a step ahead.',
                    # 'Bound a step forward.',
                    # 'Bound a step ahead.',
                    ]

jump_backward_instructions = ['Jump a step backward.',
                    # 'Jump a step back.',
                    # 'Leap a step backward.',
                    # 'Leap a step back.',
                    # 'Hop a step backward.',
                    # 'Hop a step back.',
                    # 'Spring a step backward.',
                    # 'Spring a step back.',
                    # 'Bound a step backward.',
                    # 'Bound a step back.',
                    ]

move_forward_instructions = [f'Run forward.',
                    # f'Run ahead.',
                    # f'Sprint forward.',
                    # f'Sprint ahead.',
                    # f'Rush forward.',
                    # f'Rush ahead.',
                    # f'Dash forward.',
                    # f'Dash ahead.',
                    # f'Move forward.',
                    # f'Move ahead.',
                    ]

move_backward_instructions = [f'Run backward.',
                    # f'Run back.',
                    # f'Sprint backward.',
                    # f'Sprint back.',
                    # f'Rush backward.',
                    # f'Rush back.',
                    # f'Dash backward.',
                    # f'Dash back.',
                    # f'Move backward.',
                    # f'Move back.',
                    ]

rep_jump_forward_instructions = ['Jump a step forth.',
                    'Jump one step ahead.',
                    'Leap a step forth.',
                    'Leap one step ahead.',
                    'Skip a step forth.',
                    'Skip one step ahead.',
                    'Hop a step forth.',
                    'Hop one step forward.',
                    'Spring a step forth.',
                    'Spring one step ahead.',]

rep_jump_backward_instructions = ['Jump one step backward.',
                    'Jump one step back.',
                    'Leap one step backward.',
                    'Leap one step back.',
                    'Skip one step backward.',
                    'Skip one step back.',
                    'Hop one step backward.',
                    'Hop one step back.',
                    'Spring one step backward.',
                    'Spring one step back.',]

rep_move_forward_instructions = [f'Speed forward.',
                    f'Speed ahead.',
                    f'Race forward.',
                    f'Race ahead.',
                    f'Charge forward.',
                    f'Charge ahead.',
                    f'Bolt forward.',
                    f'Bolt ahead.',
                    f'Advance forward.',
                    f'Advance ahead.',]
rep_move_backward_instructions = [f'Speed backward.',
                    f'Speed back.',
                    f'Race backward.',
                    f'Race back.',
                    f'Retreat backward.',
                    f'Retreat back.',
                    f'Hurry backward.',
                    f'Hurry back.',
                    f'Backpedal backward.',
                    f'Backpedal back.',]
move_forward_fast_instructions = [f'Move forward faster.']
move_backward_fast_instructions = [f'Move backward faster.']
move_forward_backward_instructions = [f'Move forward and slow down. Move backward.']
move_backward_forward_instructions = [f'Move backward and slow down. Move forward.']
jump_in_place_instructions = ['Jump in the original position.']
class HalfCheetahEnv(gym.Env):
    def __init__(self, task=None):
        super(HalfCheetahEnv, self).__init__()
        self.task = task
        self.env = gym.make("HalfCheetah-v2")
        self.env._max_episode_steps = 200
        self.observation_space =  spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.timestep = 0
        self.x_position = 0
        self.history_x = []
    def reset(self):
        self.timestep = 0
        obs = self.env.reset()
        self.x_position = self.env.data.qpos[0]
        self.history_x = []
        return np.concatenate(([self.x_position], obs), axis=0)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        curr_x_position = self.env.data.qpos[0]
        self.timestep += 1
        obs = np.concatenate([[curr_x_position - self.x_position], obs], axis=0)
        self.history_x.append(obs[0])
        self.x_position = curr_x_position
        reward = self.get_reward(info)
        success = self.get_success()
        info['success'] = success
        if success:
            done = True
        return obs, reward, done, info
    def get_success(self):
        success = False
        if self.task == 'jump-forward':
            if np.sum(self.history_x) > 0 and self.timestep >= 5:
                success = True
        elif self.task == 'jump-backward':
            if np.sum(self.history_x) < 0 and self.timestep >= 5:
                success = True
        elif self.task == 'rep-jump-forward':
            if np.sum(self.history_x) > 0 and self.timestep >= 5:
                success = True
        elif self.task == 'rep-jump-backward':
            if np.sum(self.history_x) < 0 and self.timestep >= 5:
                success = True
        elif self.task == 'move-forward':
            if np.sum(self.history_x) > 7 and self.timestep >= 99:
                success = True
        elif self.task == 'move-backward':
            if np.sum(self.history_x) < -7 and self.timestep >= 99:
                success = True
        elif self.task == 'rep-move-forward':
            if np.sum(self.history_x) > 7 and self.timestep >= 99:
                success = True
        elif self.task == 'rep-move-backward':
            if np.sum(self.history_x) < -7 and self.timestep >= 99:
                success = True
        elif self.task == 'move-forward-fast':
            if np.sum(self.history_x) > 8.5 and self.timestep >= 99:
                success = True
        elif self.task == 'move-backward-fast':
            if np.sum(self.history_x) < -8.5 and self.timestep >= 99:
                success = True
        elif self.task == 'move-forward-backward':
            count_negative = sum(1 for x in self.history_x if x < 0)
            count_positive = sum(1 for x in self.history_x if x > 0)
            sum_negative = sum(x for x in self.history_x if x < 0)
            sum_positive = sum(x for x in self.history_x if x > 0)
            if count_negative > 30 and count_positive > 30 and sum_negative < -3 and sum_positive > 3 and self.timestep >= 199:
                success = True
        elif self.task == 'move-backward-forward':
            count_negative = sum(1 for x in self.history_x if x < 0)
            count_positive = sum(1 for x in self.history_x if x > 0)
            sum_negative = sum(x for x in self.history_x if x < 0)
            sum_positive = sum(x for x in self.history_x if x > 0)
            if count_negative > 30 and count_positive > 30 and sum_negative < -3 and sum_positive > 3 and self.timestep >= 199:
                success = True
        elif self.task == 'jump-in-place':
            if abs(np.sum(self.history_x)) < 1.05 and self.timestep >= 15:
                success = True
        else:
            raise ValueError(f"can't find {self.task} success function")
        return success
    def get_reward(self, info):
        if self.task in forward_env_name_list:
            reward = info['reward_run'] + info['reward_ctrl']
        elif self.task in backward_env_name_list:
            reward = -info['reward_run'] + info['reward_ctrl']
        elif self.task in forward_backward_env_name_list:
            reward = abs(info['reward_run']) + info['reward_ctrl']
        elif self.task == "jump-in-place":
            reward = - abs(info['reward_run']) + info['reward_ctrl']
        else:
            raise ValueError(f"can't find {self.task} reward function")
        return reward
    def get_instructions(self):
        if self.task == 'jump-forward':
            inst = jump_backward_instructions
        elif self.task == 'jump-backward':
            inst = jump_forward_instructions
        elif self.task == 'rep-jump-forward':
            inst = rep_jump_forward_instructions
        elif self.task == 'rep-jump-backward':
            inst = rep_jump_backward_instructions
        elif self.task == 'move-forward':
            inst = move_forward_instructions
        elif self.task == 'move-backward':
            inst = move_backward_instructions
        elif self.task == 'rep-move-forward':
            inst = rep_move_forward_instructions
        elif self.task == 'rep-move-backward':
            inst = rep_move_backward_instructions
        elif self.task == 'move-forward-fast':
            inst = move_forward_fast_instructions
        elif self.task == 'move-backward-fast':
            inst = move_backward_fast_instructions
        elif self.task == 'move-forward-backward':
            inst = move_forward_backward_instructions
        elif self.task == 'move-backward-forward':
            inst = move_backward_forward_instructions
        elif self.task == 'jump-in-place':
            inst = jump_in_place_instructions
        else:
            raise ValueError(f"can't find {self.task} instructions")
        return inst

    