import os
import json
from typing import Any, Dict, List, Sequence, Union
import algo.d3rlpy as d3rlpy
import numpy as np
from tqdm import tqdm
from algo.d3rlpy.algos.qlearning import QLearningAlgoBase


class CallBack():
    def __init__(self):
        self.env_dict = None
    def add_eval_env(self, env_dict, eval_num, eval_json_save_path=None):
        self.env_dict = env_dict
        self.env_num = eval_num
        self.eval_json_save_path = eval_json_save_path
    def EvalCallback(self, agent: QLearningAlgoBase, epoch: int, total_step: int):
        for test_level, env in self.env_dict.items():
            obs = env.reset()

            traj_reward = 0
            traj_len = 0
            
            succ_list = []
            reward_list = []
            traj_len_list = []

            pbar = tqdm(total=self.env_num, desc=f'epoch {epoch} eval: {test_level}', ncols=120, leave=False)
            while True:
                action = agent.predict(obs[np.newaxis, ...])
                action = action[0]
                obs, reward, done, info = env.step(action)
                traj_reward += reward
                traj_len += 1

                if done:
                    succ_list.append(1 if info['success'] else 0)
                    reward_list.append(traj_reward)
                    traj_len_list.append(traj_len)
                    
                    pbar.update()

                    if len(succ_list) == self.env_num:
                        break
                    else:
                        obs = env.reset()
                        traj_reward = 0
                        traj_len = 0
            pbar.close()

            
            agent.logger.add_metric(f'eval/{test_level}_succ', np.mean(succ_list))
            agent.logger.add_metric(f'eval/{test_level}_reward', np.mean(reward_list))
            agent.logger.add_metric(f'eval_len/{test_level}_len', np.mean(traj_len_list))
            if self.eval_json_save_path is not None:
                if os.path.exists(self.eval_json_save_path):
                    with open(self.eval_json_save_path, 'r') as f:
                        data = json.load(f)
                else:
                    data = {}
                data: Dict[str, Dict[int, float]]
                
                if test_level in data.keys():
                    data[test_level][epoch] = np.mean(succ_list)
                else:
                    data[test_level] = {epoch: np.mean(succ_list)}
                
                with open(self.eval_json_save_path, 'w') as f:
                    json.dump(data, f)