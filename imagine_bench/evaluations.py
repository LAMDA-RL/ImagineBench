from typing import Dict

import numpy as np
from tqdm import tqdm

from envs import RIMAROEnv
from algo.d3rlpy.algos.qlearning import QLearningAlgoBase


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
