import random
from collections import Counter
from typing import Any, Dict, List, Union

import gym
import gym.spaces
import numpy as np
import gym.spaces.box
from gymnasium import Wrapper, spaces
from minigrid.core.world_object import Wall, WorldObj
from minigrid.envs.babyai.core.verifier import ObjDesc, GoToInstr
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX, OBJECT_TO_IDX, IDX_TO_COLOR

from grid_utils import my_heuristic, get_target_pos
from grid_utils import OBJ_TYPES, CUSTOM_ACTION_TO_NEED_DIR


class MyGrid(RoomGridLevel):
    def __init__(self, room_size:int = 8, max_steps:int | None = None):
        super().__init__(num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps)
        self.render_mode = 'rgb_array'

        self.obj_color: str = None
        self.obj_type: str = None

        self.task_str: str
        self.instruction: List[str]
        self.novel_instruction: List[str]

        self.distance_start: float
        self.base_penalty = self.room_size * 3
        self.obj_type_color_dict = {obj_type: '' for obj_type in OBJ_TYPES}
        self.obj_last_pos_dict: Dict[str, np.ndarray] = {obj_type: None for obj_type in OBJ_TYPES}

        self.env_name: str
    
    def custom_step(self, action: int) -> tuple[Union[Dict[str, Any], np.ndarray], float, bool, bool, dict]:
        # modified from minigrid.minigrid_env.MiniGridEnv.step
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        current_cell = self.grid.get(*self.agent_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or (fwd_cell.type in ['ball', 'box', 'key']):  ## change to everything can be overlapped
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if current_cell and current_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = current_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(self.agent_pos[0], self.agent_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not current_cell and self.carrying:
                self.grid.set(self.agent_pos[0], self.agent_pos[1], self.carrying)
                self.carrying.cur_pos = self.agent_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            ## change to try to toggle where cell near agent
            near_pos_list = []
            near_pos_list.append((self.agent_pos[0]-1, self.agent_pos[1]))
            near_pos_list.append((self.agent_pos[0]+1, self.agent_pos[1]))
            near_pos_list.append((self.agent_pos[0], self.agent_pos[1]-1))
            near_pos_list.append((self.agent_pos[0], self.agent_pos[1]+1))
            for near_pos in near_pos_list:
                near_cell = self.grid.get(*near_pos)
                if (near_cell is not None) and (near_cell.type == 'door'):
                    near_cell.toggle(self, near_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
    
    def my_state(self) -> np.ndarray:
        """
        only for env.predict, not for train
        """
        grid_encode = self.grid.encode()

        grid_encode[self.agent_pos[0], self.agent_pos[1], 2] = OBJECT_TO_IDX['agent'] * (self.agent_dir+1)
        carrying_pos = (0, 0)  # encode carrying in unused corner
        if self.carrying is not None:
            # type and color
            grid_encode[carrying_pos[0], carrying_pos[1], :2] = self.carrying.encode()[:2]
        else:
            # set empty cell to uniqe unseen
            grid_encode[carrying_pos[0], carrying_pos[1], 0] = OBJECT_TO_IDX['unseen']
        
        state_full = grid_encode.transpose(1, 0, 2)  # transpose to match visualization

        state = state_full[:, :self.room_size]  # remove unused rows

        return state
    
    def my_obs(self) -> np.ndarray:
        obs_list = []
        for obj_type, obj_color in self.obj_type_color_dict.items():
            obj_pos = get_target_pos(state=self.my_state(), target_color=obj_color, target_type=obj_type)
            assert obj_pos is not None
            obs_list.extend([COLOR_TO_IDX[obj_color], obj_pos[0], obj_pos[1]])
            if obj_type == 'door':
                obs_list.append(self.grid.get(*np.flip(obj_pos)).encode()[-1])  # add if door open, reverse to match visualization
        obs_list.extend([self.agent_pos[1], self.agent_pos[0]])  # reverse to match visualization
        if self.carrying is not None:
            carrying_list = list(self.carrying.encode()[:2])
        else:
            carrying_list = [OBJECT_TO_IDX['unseen'], 0]
        obs_list.extend(carrying_list)

        # [ball_color, ball_x, ball_y,
        # box_color, box_x, box_y,
        # key_color, key_x, key_y,
        # door_color, door_x, door_y, door_state
        # agent_x, agent_y,
        # carrying_type, carrying_color]  # not used
        return np.array(obs_list, dtype=np.uint8)
    
    def predict_goto(self, agent_pos: np.ndarray, target_pos: np.ndarray) -> int:
        if agent_pos[0] < target_pos[0]:
            return self.actions.done  # go down
        elif agent_pos[0] > target_pos[0]:
            return self.actions.forward  # go up
        elif agent_pos[1] < target_pos[1]:
            return self.actions.right  # go right
        elif agent_pos[1] > target_pos[1]:
            return self.actions.left  # go left
        else:
            return self.actions.right  # for door, in case bad inialization
    
    def estimated_distance(self, state: np.ndarray) -> float:
        """
        only for one obj env
        """
        target_pos = get_target_pos(state=state, target_color=self.obj_color, target_type=self.obj_type)
        if target_pos is None:
            target_pos = self.obj_last_pos_dict[self.obj_type]
            assert target_pos is not None
        return my_heuristic(target_pos=target_pos, state=state)

    def dense_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """
        compute dense reward based on the distance to the target, only for one obj env
        """
        distance = self.estimated_distance(state=state)
        distance_next = self.estimated_distance(state=next_state)
        return (distance - distance_next) / (self.distance_start + 1)

    def verify(self) -> bool:
        raise NotImplementedError

    def predict(self) -> int:
        raise NotImplementedError

    def record_last_pos(self) -> None:
        for obj_type, obj_color in self.obj_type_color_dict.items():
            obj_pos = get_target_pos(state=self.my_state(), target_color=obj_color, target_type=obj_type)
            if obj_pos is not None:
                self.obj_last_pos_dict[obj_type] = obj_pos

    def build_from_my_obs(self, my_obs: np.ndarray) -> bool:
        # get empty encode
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                if i in [0, self.room_size-1, self.room_size*2-2] or j in [0, self.room_size-1]:
                    self.grid.set(i, j, Wall())
                else:
                    self.grid.set(i, j, None)  # all empty except wall
        self.carrying = None

        # ball, box, key, door
        # check pos overlap
        pos_list = []
        for i in range(0, (1+2)*len(OBJ_TYPES), 1+2):
            obj_x = my_obs[i+1]
            obj_y = my_obs[i+2]
            pos_list.append((obj_x, obj_y))
        pos_counter = Counter(pos_list)
        if pos_counter.most_common(1)[0][1] > 1:
            pos_overlap = True
        else:
            pos_overlap = False
        
        # build
        for i in range(0, (1+2)*len(OBJ_TYPES), 1+2):
            obj_color_idx = my_obs[i]
            if not pos_overlap:
                obj_x = my_obs[i+1]
                obj_y = my_obs[i+2]
            else:
                obj_x = self.obj_last_pos_dict[list(self.obj_type_color_dict.keys())[i//3]][0]
                obj_y = self.obj_last_pos_dict[list(self.obj_type_color_dict.keys())[i//3]][1]
            cell_state = 0 if i // 3 != 3 else my_obs[i+3]
            if not ((obj_x == 0) and (obj_y == 0)):
                self.grid.set(
                    obj_y, obj_x, WorldObj.decode(OBJECT_TO_IDX[list(self.obj_type_color_dict.keys())[i//3]], obj_color_idx, cell_state)
                    )  # reverse to match visualization
            else:
                self.carrying = WorldObj.decode(OBJECT_TO_IDX[list(self.obj_type_color_dict.keys())[i//3]], obj_color_idx, 0)
            self.obj_type_color_dict[list(self.obj_type_color_dict.keys())[i//3]] = IDX_TO_COLOR[obj_color_idx]

        self.agent_pos = (my_obs[-3], my_obs[-4])  # reverse to match visualization
        self.agent_dir = 3

        self.record_last_pos()

        return pos_overlap

    def gen_mission(self):
        self.place_agent(0, 0)

        for obj_type in self.obj_type_color_dict.keys():
            if self.obj_type_color_dict[obj_type] == '':
                self.obj_type_color_dict[obj_type] = random.choice(COLOR_NAMES)
            if obj_type == 'door':
                _ = self.add_door(0, 0, color=self.obj_type_color_dict[obj_type], locked=False)
            else:
                obj, _ = self.add_object(0, 0, obj_type, self.obj_type_color_dict[obj_type])
        
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))  # dummy instruction, won't be used

    def get_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        if self.verify():
            reward = self._reward()
        else:
            reward = 0.0
        
        if reward < 0.1:
            reward += self.dense_reward(state=state, next_state=next_state)
        
        return reward


class DenseRewardWrapper(Wrapper):
    def __init__(self, env: MyGrid):
        Wrapper.__init__(self, env)

    def step(self, action: int) -> tuple[Union[Dict[str, Any], np.ndarray], float, bool, bool, dict]:
        assert isinstance(self.env.unwrapped, MyGrid)
        state = self.env.unwrapped.my_state()
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward < 0.1:
            reward += self.env.unwrapped.dense_reward(state=state, next_state=self.env.unwrapped.my_state())
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> tuple[Union[Dict[str, Any], np.ndarray], dict]:
        assert isinstance(self.env.unwrapped, MyGrid)
        obs, info = self.env.reset(**kwargs)
        self.env.unwrapped.distance_start = self.env.unwrapped.estimated_distance(state=self.env.unwrapped.my_state())
        return obs, info


class EncodeWrapper(Wrapper):
    def __init__(self, env: MyGrid):
        Wrapper.__init__(self, env)

        assert isinstance(self.observation_space, spaces.Box)

        new_image_space = spaces.Box(
            low=self.observation_space.low.flatten()[0],
            high=self.observation_space.high.flatten()[0],
            shape=((1+2)*(len(OBJ_TYPES)-1)+(1+2+1)+2+2, ),  # 17
            dtype=self.observation_space.dtype,
        )

        self.observation_space = new_image_space

    def observation(self, obs: Union[np.ndarray, Dict[str, Any]]) -> np.ndarray:
        assert isinstance(self.env.unwrapped, MyGrid)
        return self.env.unwrapped.my_obs()
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert isinstance(self.env.unwrapped, MyGrid)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['obs_raw'] = obs
        return self.env.unwrapped.my_obs(), reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        assert isinstance(self.env.unwrapped, MyGrid)
        obs, info = self.env.reset(**kwargs)
        info['obs_raw'] = obs
        return self.env.unwrapped.my_obs(), info


class ActionWrapper(Wrapper):
    def __init__(self, env: MyGrid):
        Wrapper.__init__(self, env)
    
    def step(self, action: int) -> tuple[Union[Dict[str, Any], np.ndarray], float, bool, bool, Dict[str, Any]]:
        # modified from minigrid.envs.babyai.core.roomgrid_level.RoomGridLevel.step
        assert isinstance(self.env.unwrapped, MyGrid)
        if action in [
            self.env.unwrapped.actions.left, self.env.unwrapped.actions.right,
            self.env.unwrapped.actions.forward, self.env.unwrapped.actions.done
            ]:
            # first turn to needed direction
            self.env.unwrapped.agent_dir = CUSTOM_ACTION_TO_NEED_DIR[action]
            # then step forward
            obs, reward, terminated, truncated, info = self.env.unwrapped.custom_step(action=self.env.unwrapped.actions.forward)
        elif action in [self.env.unwrapped.actions.pickup, self.env.unwrapped.actions.drop, self.env.unwrapped.actions.toggle]:
            obs, reward, terminated, truncated, info = self.env.unwrapped.custom_step(action=action)
        else:
            raise NotImplementedError
        
        if action == self.env.unwrapped.actions.drop:
            self.env.unwrapped.update_objs_poss()

        # If we've successfully completed the mission
        if self.env.unwrapped.verify():
            terminated = True
            reward = self.env.unwrapped._reward()

        return obs, reward, terminated, truncated, info


class LanguageWrapper(Wrapper):
    def __init__(self, env_list: List[MyGrid], inst_encode_path: str, level: str, use_gym: bool = False):
        self.env_list = env_list
        self.env_idx: int = 0
        Wrapper.__init__(self, self.env_list[self.env_idx])
        self.env = self.env_list[self.env_idx]
        self.level = level

        try:
            inst2encode = np.load(inst_encode_path, allow_pickle=True).item()
        except:
            with open(inst_encode_path, "rb") as f:
                inst2encode = np.load(f, allow_pickle=True)
        self.inst2encode: Dict[str, np.ndarray] = inst2encode

        assert isinstance(self.observation_space, spaces.Box)

        if not use_gym:
            space = spaces  # gymnasium
        else:
            space = gym.spaces

        assert isinstance(self.env_list[0].unwrapped, MyGrid)
        new_image_space = space.Box(
            low=gym.spaces.box.get_inf(self.inst2encode[self.env_list[0].unwrapped.task_str].dtype, sign='-'),
            high=gym.spaces.box.get_inf(self.inst2encode[self.env_list[0].unwrapped.task_str].dtype, sign='+'),
            shape=(self.observation_space.shape[0]+768, ),  # 17+768=785
            dtype=self.inst2encode[self.env_list[0].unwrapped.task_str].dtype,
        )

        self.observation_space = new_image_space

        assert isinstance(self.action_space, spaces.Discrete)
        
        new_action_space = space.Discrete(
            n=self.action_space.n,
            start=self.action_space.start,
        )

        self.action_space = new_action_space

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs: np.ndarray
        obs = obs.astype(self.inst_encode.dtype)
        obs = np.concatenate([obs, self.inst_encode], axis=0)
        return obs
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs: np.ndarray
        obs = obs.astype(self.inst_encode.dtype)
        obs = np.concatenate([obs, self.inst_encode], axis=0)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        self.env_idx += 1
        self.env_idx %= len(self.env_list)
        self.env = self.env_list[self.env_idx]

        obs, info = self.env.reset(**kwargs)
        obs: np.ndarray

        assert isinstance(self.env.unwrapped, MyGrid)
        if self.level != 'rephrase':
            self.inst_encode = self.inst2encode[self.env.unwrapped.task_str]
        else:
            self.inst_encode = self.inst2encode[self.env.unwrapped.novel_instruction[0]]
        obs = obs.astype(self.inst_encode.dtype)
        obs = np.concatenate([obs, self.inst_encode], axis=0)
        return obs, info


class MyWrapper(Wrapper):
    def __init__(self, env: MyGrid):
        Wrapper.__init__(self, env)

    def reset(self, **kwargs) -> tuple[Union[Dict[str, Any], np.ndarray], dict]:
        assert isinstance(self.env.unwrapped, MyGrid)
        obs, info = self.env.reset(**kwargs)

        self.env.unwrapped.record_last_pos()
        
        return obs, info
