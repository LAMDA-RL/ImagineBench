from typing import Dict

import numpy as np
from tqdm import tqdm

from envs import RIMAROEnv
from algo.d3rlpy.algos.qlearning import QLearningAlgoBase
from offlinerl.evaluation import CallBackFunction
class EvalCallBackFunction(CallBackFunction):
    def initialize(self, env_dict, number_of_runs: int = 10, writer=None):
        self.env_dict = env_dict
        self.env_list = [env for env in env_dict.values()]
        self.level_list = [level for level in env_dict.keys()]
        self.is_initialized = True
        self.number_of_runs = number_of_runs
        self.writer = writer
        self.call_num = 0
    
    def __call__(self, policy):
        assert self.is_initialized, "`initialize` should be called before callback."
        self.call_num += 1
        # policy = deepcopy(policy).to('cpu')
        eval_res = {}
        for i in range(len(self.env_list)):
            env = self.env_list[i]
            obs, _ = env.reset()
            reward_list = []
            length_list = []
            ep_reward = 0
            ep_length = 0
            success_num = 0
            while True:
                action = policy.get_action(obs.reshape(-1, env.observation_space.shape[0])).reshape(env.action_space.shape[0])
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1
                if info['is_success']:
                    success_num += 1
                if done:
                    reward_list.append(ep_reward)
                    length_list.append(ep_length)
            
                    if len(reward_list) == self.number_of_runs:
                       break
                    else:
                        obs, _ = env.reset()
                        # print(self.env.env_idx)
                        ep_reward = 0
                        ep_length = 0
            eval_res[f'reward_{self.level_list[i]}'] = np.mean(reward_list)
            eval_res[f'length_{self.level_list[i]}'] = np.mean(length_list)
            eval_res[f'success_{self.level_list[i]}'] = success_num / self.number_of_runs
            for key in eval_res.keys():
                self.writer.add_scalar(key, eval_res[key], self.call_num)
        return eval_res

class CallBack():
    def __init__(self):
        self.env_dict: Dict[str, RIMAROEnv] = None
    
    def add_eval_env(self, env_dict: Dict[str, RIMAROEnv], eval_num: int) -> None:
        self.env_dict = env_dict
        self.env_num = eval_num
    
    def EvalCallback(self, agent: QLearningAlgoBase, epoch: int, total_step: int) -> None:
        for test_level, env in self.env_dict.items():
            obs, _ = env.reset()

            traj_reward = 0
            traj_len = 0
            
            succ_list = []
            reward_list = []
            traj_len_list = []

            pbar = tqdm(total=self.env_num, desc=f'epoch {epoch} eval: {test_level}', ncols=120, leave=False)
            while True:
                action = agent.predict(obs[np.newaxis, ...])
                action = action[0]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                traj_reward += reward
                traj_len += 1

                if done:
                    succ_list.append(1 if info['is_success'] else 0)
                    reward_list.append(traj_reward)
                    traj_len_list.append(traj_len)
                    
                    pbar.update()

                    if len(succ_list) == self.env_num:
                        break
                    else:
                        obs, _ = env.reset()
                        traj_reward = 0
                        traj_len = 0
            pbar.close()

            
            agent.logger.add_metric(f'eval/{test_level}_succ', np.mean(succ_list))
            agent.logger.add_metric(f'eval/{test_level}_reward', np.mean(reward_list))
            agent.logger.add_metric(f'eval_len/{test_level}_len', np.mean(traj_len_list))
