import numpy as np
from minigrid.core.world_object import Door
from minigrid.envs.babyai.core.verifier import pos_next_to
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from base import MyGrid
from grid_utils import my_heuristic, get_target_pos, get_agent_pos


class OpenLockEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None):
        super().__init__(room_size=room_size, max_steps=max_steps)

        self.task_str = f'pick up the key, then open the door'

        self.instruction = [self.task_str for _ in range(10)]

        self.env_name = 'open_lock'

    def gen_mission(self):
        super().gen_mission()
        
        door_pos = get_target_pos(state=self.my_state(), target_color=self.obj_type_color_dict['door'], target_type='door')
        door: Door = self.grid.get(*np.flip(door_pos))
        door.color = self.obj_type_color_dict['key']
        self.obj_type_color_dict['door'] = door.color
        door.is_locked = True
        
        self.obj_color = door.color

    def verify(self) -> bool:
        state = self.my_state()
        target_pos = get_target_pos(state=state, target_type='door', target_color=self.obj_color)
        if target_pos is None:
            return False  # can not find target
        
        if state[target_pos[0], target_pos[1], 2] == STATE_TO_IDX['open']:
            return True
        else:
            return False
    
    def estimated_distance(self, state: np.ndarray) -> float:
        target1_pos = get_target_pos(state=state, target_color=self.obj_color, target_type='key')
        target2_pos = get_target_pos(state=state, target_color=self.obj_type_color_dict['door'], target_type='door')
        if target1_pos is None:
            target1_pos = self.obj_last_pos_dict['key']
            assert target1_pos is not None
        if target2_pos is None:
            target2_pos = self.obj_last_pos_dict['door']
            assert target2_pos is not None
        target2_pos += np.array([0, -1])
        
        if state[0, 0, 0] == OBJECT_TO_IDX['unseen']:
            # no pick up yet
            distance1 = my_heuristic(target_pos=target1_pos, state=state)
            distance2 = my_heuristic(target_pos=target2_pos, state=state, start_pos=target1_pos)
            penalty = self.base_penalty
        elif not ((state[0, 0, 0] == OBJECT_TO_IDX['key']) and (state[0, 0, 1] == COLOR_TO_IDX[self.obj_color])):
            # pick up wrong object
            distance1 = 0.0
            distance2 = 0.0
            penalty = self.base_penalty * 3
        else:
            # key already picked up
            distance1 = 0.0
            distance2 = my_heuristic(target_pos=target2_pos, state=state)
            penalty = 0.0
        return distance1 + distance2 + penalty


class PutLineEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None):
        super().__init__(room_size=room_size, max_steps=max_steps)

        self.task_str = f'put the three items in a line'

        self.instruction = [self.task_str for _ in range(10)]

        self.env_name = 'put_line'

    def verify(self):
        state = self.my_state()
        target_pos_list = []
        for target_type, target_color in self.obj_type_color_dict.items():
            if target_type != 'door':
                target_pos = get_target_pos(state=state, target_color=target_color, target_type=target_type)
                if target_pos is None:
                    return False
                else:
                    target_pos_list.append(target_pos)
            
        if target_pos_list[0][0] == target_pos_list[1][0] == target_pos_list[2][0]:
            return True
        elif target_pos_list[0][1] == target_pos_list[1][1] == target_pos_list[2][1]:
            return True
        else:
            return False
    
    def estimated_distance(self, state: np.ndarray) -> float:
        target_pos_list = []
        for target_type, target_color in self.obj_type_color_dict.items():
            if target_type != 'door':
                target_pos = get_target_pos(state=state, target_color=target_color, target_type=target_type)
                if target_pos is None:
                    target_pos = self.obj_last_pos_dict[target_type]
                if (target_pos[0] == 0) and (target_pos[1] == 0):
                    target_pos = get_agent_pos(state=state)
                target_pos_list.append(target_pos)
        distance1 = abs(target_pos_list[0][0] - target_pos_list[1][0]) + \
            abs(target_pos_list[1][0] - target_pos_list[2][0]) + \
            abs(target_pos_list[2][0] - target_pos_list[0][0])
        distance2 = abs(target_pos_list[0][1] - target_pos_list[1][1]) + \
            abs(target_pos_list[1][1] - target_pos_list[2][1]) + \
            abs(target_pos_list[2][1] - target_pos_list[0][1])
        return float(min(distance1, distance2))


class PutPileEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None):
        super().__init__(room_size=room_size, max_steps=max_steps)

        self.obj_type = None
        self.obj_color = None
        
        self.task_str = f'gather the three items into a pile'

        self.instruction = [self.task_str for _ in range(10)]

        self.env_name = 'put_pile'
    
    def verify(self):
        state = self.my_state()
        target_pos_list = []
        for target_type, target_color in self.obj_type_color_dict.items():
            if target_type != 'door':
                target_pos = get_target_pos(state=state, target_color=target_color, target_type=target_type)
                if target_pos is None:
                    return False
                else:
                    target_pos_list.append(target_pos)

        pile_count = 0
        if pos_next_to(pos_a=target_pos_list[0], pos_b=target_pos_list[1]):
            pile_count += 1
        if pos_next_to(pos_a=target_pos_list[1], pos_b=target_pos_list[2]):
            pile_count += 1
        if pos_next_to(pos_a=target_pos_list[2], pos_b=target_pos_list[0]):
            pile_count += 1
        
        if pile_count >= 2:
            if target_pos_list[0][0] == target_pos_list[1][0] == target_pos_list[2][0]:
                return False
            elif target_pos_list[0][1] == target_pos_list[1][1] == target_pos_list[2][1]:
                return False
            else:
                return True
        else:
            return False
    
    def estimated_distance(self, state: np.ndarray) -> float:
        target_pos_list = []
        for target_type, target_color in self.obj_type_color_dict.items():
            if target_type != 'door':
                target_pos = get_target_pos(state=state, target_color=target_color, target_type=target_type)
                if target_pos is None:
                    target_pos = self.obj_last_pos_dict[target_type]
                if (target_pos[0] == 0) and (target_pos[1] == 0):
                    target_pos = get_agent_pos(state=state)
                target_pos_list.append(target_pos)
        distance1 = abs(target_pos_list[0][0] - target_pos_list[1][0]) + abs(target_pos_list[0][1] - target_pos_list[1][1])
        distance2 = abs(target_pos_list[1][0] - target_pos_list[2][0]) + abs(target_pos_list[1][1] - target_pos_list[2][1])
        distance3 = abs(target_pos_list[2][0] - target_pos_list[0][0]) + abs(target_pos_list[2][1] - target_pos_list[0][1])
        return float(distance1 + distance2 + distance3)
