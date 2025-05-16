import random

import numpy as np
from minigrid.core.constants import STATE_TO_IDX

from base import MyGrid
from grid_utils import my_heuristic, get_target_pos, get_agent_pos


class OpenGoEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None):
        super().__init__(room_size=room_size, max_steps=max_steps)

        self.task_str = f'open the door, then goto any object'

        self.instruction = [self.task_str for _ in range(10)]

        self.env_name = 'open_go'

    def verify(self) -> bool:
        state = self.my_state()
        door_pos = get_target_pos(state=state, target_color=self.obj_type_color_dict['door'], target_type='door')
        if door_pos is None:
            return False
        if state[door_pos[0], door_pos[1], 2] == STATE_TO_IDX['open']:
            agent_pos = get_agent_pos(state=state)
            for target_type, target_color in self.obj_type_color_dict.items():
                if target_type != 'door':
                    target_pos = get_target_pos(state=state, target_color=target_color, target_type=target_type)
                    if target_pos is None:
                        target_pos = self.obj_last_pos_dict[target_type]
                    if np.array_equal(agent_pos, target_pos):
                        return True
        return False

    def estimated_distance(self, state: np.ndarray) -> float:
        target_pos_list = []
        for target_type, target_color in self.obj_type_color_dict.items():
            if target_type != 'door':
                target_pos = get_target_pos(state=state, target_color=target_color, target_type=target_type)
                if target_pos is None:
                    target_pos = self.obj_last_pos_dict[target_type]
                target_pos_list.append(target_pos)

        door_pos = get_target_pos(state=state, target_color=self.obj_type_color_dict['door'], target_type='door')
        if door_pos is None:
            door_pos = self.obj_last_pos_dict['door']

        if state[door_pos[0], door_pos[1], 2] != STATE_TO_IDX['open']:
            intermediate_pos = door_pos + np.array([0, -1])
            distance1 = my_heuristic(target_pos=intermediate_pos, state=state)
            distance_list = [my_heuristic(target_pos=target_pos, state=state, start_pos=intermediate_pos) for target_pos in target_pos_list]
            distance2 = min(distance_list)
            penalty = self.base_penalty
        else:
            distance1 = 0.0
            distance_list = [my_heuristic(target_pos=target_pos, state=state) for target_pos in target_pos_list]
            distance2 = min(distance_list)
            penalty = 0.0
            
        return distance1 + distance2 + penalty


class OpenPickEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None):
        super().__init__(room_size=room_size, max_steps=max_steps)

        self.task_str = f'open the door, then pick up any object'

        self.instruction = [self.task_str for _ in range(10)]

        self.env_name = 'open_pick'

    def verify(self) -> bool:
        state = self.my_state()
        door_pos = get_target_pos(state=state, target_color=self.obj_type_color_dict['door'], target_type='door')
        if door_pos is not None:
            return False
        if state[door_pos[0], door_pos[1], 2] == STATE_TO_IDX['open']:
            if self.carrying is not None:
                return True
        return False

    def estimated_distance(self, state: np.ndarray) -> float:
        target_pos_list = []
        for target_type, target_color in self.obj_type_color_dict.items():
            if target_type != 'door':
                target_pos = get_target_pos(state=state, target_color=target_color, target_type=target_type)
                if target_pos is None:
                    target_pos = self.obj_last_pos_dict[target_type]
                target_pos_list.append(target_pos)

        door_pos = get_target_pos(state=state, target_color=self.obj_type_color_dict['door'], target_type='door')
        if door_pos is None:
            door_pos = self.obj_last_pos_dict['door']
        if state[door_pos[0], door_pos[1], 2] != STATE_TO_IDX['open']:
            intermediate_pos = door_pos + np.array([0, -1])
            distance1 = my_heuristic(target_pos=intermediate_pos, state=state)
            distance_list = [my_heuristic(target_pos=target_pos, state=state, start_pos=intermediate_pos) for target_pos in target_pos_list]
            distance2 = min(distance_list)
            penalty = self.base_penalty
        else:
            distance1 = 0.0
            distance_list = [my_heuristic(target_pos=target_pos, state=state) for target_pos in target_pos_list]
            distance2 = min(distance_list)
            penalty = 0.0
            
        return distance1 + distance2 + penalty


class GoWallEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None):
        super().__init__(room_size=room_size, max_steps=max_steps)

        self.task_str = f'goto the side of the wall'

        self.instruction = [self.task_str for _ in range(10)]

        self.agent_pos_choices = [3, 4]

        self.env_name = 'go_wall'
    
    def gen_mission(self):
        super().gen_mission()

        while True:
            self.agent_pos = (random.choice(self.agent_pos_choices), random.choice(self.agent_pos_choices))
            start_cell = self.grid.get(*self.agent_pos)
            if (start_cell is None) or start_cell.can_overlap():
                break
    
    def verify(self) -> bool:
        target_list = [1, self.room_size - 2]
        if (self.agent_pos[0] in target_list) or (self.agent_pos[1] in target_list):
            return True
        else:
            return False
    
    def estimated_distance(self, state: np.ndarray) -> float:
        target_list = [1, self.room_size - 2]
        distance_list = []
        for current in get_agent_pos(state=state):
            distance_list.extend([abs(current - target) for target in target_list])
        return float(min(distance_list))


class GoCenterEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None):
        super().__init__(room_size=room_size, max_steps=max_steps)

        self.task_str = f'goto the center of the room'

        self.instruction = [self.task_str for _ in range(10)]

        self.target_pos_list = [np.array([3, 3]), np.array([3, 4]), np.array([4, 3]), np.array([4, 4])]
        self.agent_pos_choices = [1, 2, 5, 6]

        self.env_name = 'go_center'
    
    def gen_mission(self):
        super().gen_mission()

        while True:
            self.agent_pos = (random.choice(self.agent_pos_choices), random.choice(self.agent_pos_choices))
            start_cell = self.grid.get(*self.agent_pos)
            if (start_cell is None) or start_cell.can_overlap():
                break

    def verify(self) -> bool:
        agent_pos = get_agent_pos(state=self.my_state())
        for target_pos in self.target_pos_list:
            if np.array_equal(agent_pos, target_pos):
                return True
        return False
    
    def estimated_distance(self, state: np.ndarray) -> float:
        distance_list = [my_heuristic(target_pos=target_pos, state=state) for target_pos in self.target_pos_list]
        distance = min(distance_list)
        return distance
