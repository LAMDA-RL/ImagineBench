import random
from typing import List, Tuple

import numpy as np
from gymnasium import Wrapper, spaces
from minigrid.core.world_object import Wall, WorldObj, Door
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, GoToInstr, OpenInstr, PickupInstr, PutNextInstr, BeforeInstr, pos_next_to

from envs.babyai.grid_utils import IDX_TO_DIR, OBJ_TYPES
from envs.babyai.grid_utils import find_path_to, my_heuristic, get_agent_pos, get_target_pos


class MyGrid(RoomGridLevel):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3):
        super().__init__(num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps)
        self.num_dists = num_dists
        self.render_mode = 'rgb_array'

        self.task_str: str
        self.instruction: List[str]
        self.novel_instruction: List[str]

        self.future_actions: List = None
        self.distance0: float = None

        self.instr_a_done = False

        self.obj_type = None
        self.obj_color = None
    
    def my_encode(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        only for env.predict, not for train
        """
        grid_encode = self.grid.encode()  # 环境state

        grid_encode[self.agent_pos[0], self.agent_pos[1], 2] = OBJECT_TO_IDX['agent'] * (self.agent_dir+1)  # agent位置的第三维代表agent的方向

        if self.carrying is not None:
            # agent位置的前两维代表agent携带的物品的类型和颜色（和放在地上的物品的形式统一）
            grid_encode[self.agent_pos[0], self.agent_pos[1], :2] = self.carrying.encode()[:2]
        else:
            # 没物品设为和其他位置empty不同的unseen
            grid_encode[self.agent_pos[0], self.agent_pos[1], 0] = OBJECT_TO_IDX['unseen']
        
        state_full = grid_encode.transpose(1, 0, 2)  # 转置，和render结果保持一致

        state = state_full[:, :self.room_size]  # 去掉无用房间

        return state, state_full

    def my_state(self) -> Tuple[np.ndarray, np.ndarray]:
        grid_encode = self.grid.encode()  # 环境state

        grid_encode[self.agent_pos[0], self.agent_pos[1], 0] = OBJECT_TO_IDX['agent']  # agent位置的第一维代表agent的type
        grid_encode[self.agent_pos[0], self.agent_pos[1], 1] = COLOR_TO_IDX['red']  # agent位置的第三维代表agent的红色
        grid_encode[self.agent_pos[0], self.agent_pos[1], 2] = self.agent_dir  # agent位置的第三维代表agent的方向
        
        state_full = grid_encode.transpose(1, 0, 2)  # 转置，和render结果保持一致

        state = state_full[1:-1, 1:self.room_size]  # 去掉无用部分

        state_f = state.reshape(-1, 3)

        carry_state = np.zeros(shape=(1, 3), dtype=state_f.dtype)
        if self.carrying is not None:
            carry_state[:, :2] = self.carrying.encode()[:2]
    
        state_f = np.concatenate([state_f, carry_state], axis=0)
        state_f = state_f.flatten()

        return state_f, state
    
    def predict_base(self, state: np.ndarray, task:str, target_type:str = None, target_color:str = None, target_position:np.ndarray = None) -> int:
        if (target_type is not None) and (target_color is not None):
            target_pos = get_target_pos(state=state, target_type=target_type, target_color=target_color)
            if (target_position is not None) and (not np.array_equal(target_pos, target_position)):
                raise NotImplementedError
        elif (target_type is None) and (target_color is None) and (target_position is not None):
            target_pos = target_position
        else:
            raise NotImplementedError

        distance = int(my_heuristic(target_pos=target_pos, state=state))
        if distance > 0:
            if self.future_actions is None:
                self.future_actions = find_path_to(target_pos=target_pos, state=state)
            else:
                self.future_actions.pop(0)
            return self.future_actions[0]
        elif distance == 0:
            if task == 'goto':
                return 0
            agent_pos = get_agent_pos(state=state)
            pos_diff = target_pos - agent_pos
            if np.array_equal(pos_diff, np.array([0, 1])):
                need_dir = 'right'
            elif np.array_equal(pos_diff, np.array([0, -1])):
                need_dir = 'left'
            elif np.array_equal(pos_diff, np.array([1, 0])):
                need_dir = 'down'
            elif np.array_equal(pos_diff, np.array([-1, 0])):
                need_dir = 'up'
            else:
                raise NotImplementedError
            
            if need_dir == IDX_TO_DIR[state[agent_pos[0], agent_pos[1], 2]]:
                if task == 'pickup':
                    self.future_actions = None  # for task: putnext
                    return 3
                elif task == 'open':
                    return 5
                else:
                    raise NotImplementedError
            else:
                return 0
        else:
            raise NotImplementedError

    def predict(self, state: np.ndarray = None) -> int:
        raise NotImplementedError

    def imagine_distance(self, state: np.ndarray, target_type:str, target_color:str) -> float:
        target_pos = get_target_pos(state=state, target_type=target_type, target_color=target_color)
        return my_heuristic(target_pos=target_pos, state=state)

    def dense_reward(self, state: np.ndarray, next_state: np.ndarray, target_type:str, target_color:str) -> float:
        distance = self.imagine_distance(state=state, target_type=target_type, target_color=target_color)
        distance_next = self.imagine_distance(state=next_state, target_type=target_type, target_color=target_color)
        return (distance - distance_next) / (self.distance0 + 1)

    def build_from_state(self, state: np.ndarray) -> None:
        # get empty encode
        for i in range(self.grid.width):
                for j in range(self.grid.height):
                    if i in [0, self.room_size-1, self.room_size*2-2] or j in [0, self.room_size-1]:
                        self.unwrapped.grid.set(i, j, Wall())
                    else:
                        self.unwrapped.grid.set(i, j, None)  # 除了墙外全空
        encode_local = self.grid.encode().transpose(1, 0, 2)

        # process stata
        state_encode = state.reshape(-1, 3)
        self.unwrapped.carrying = WorldObj.decode(*state_encode[-1])
        state_temp = state_encode[:-1]
        state_temp = state_temp.reshape(self.room_size-2, self.room_size-1, 3)
        encode_local[1:-1, 1:self.room_size] = state_temp

        for i in range(encode_local.shape[0]):
            for j in range(encode_local.shape[1]):
                if encode_local[i, j, 0] == OBJECT_TO_IDX['agent']:
                    self.unwrapped.agent_pos = (j, i)  # swap i,j because of transpose
                    encode_local[i, j, 0] = OBJECT_TO_IDX['empty']
                    self.unwrapped.agent_dir = encode_local[i, j, 2]
                    encode_local[i, j, 2] = 0
        
        encode_local = encode_local.transpose(1, 0, 2)
        self.unwrapped.grid = self.grid.decode(array=encode_local)[0]

    def verify(self) -> bool:
        raise NotImplementedError
    
    def get_reward(self) -> float:
        if self.verify():
            return 1.0
        else:
            return 0.0


class GoToEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3, obj_type:str = OBJ_TYPES[0], obj_color:str = COLOR_NAMES[0]):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.obj_type = obj_type
        self.obj_color = obj_color
        
        self.task_str = f'go to the {self.obj_color} {self.obj_type}'

        self.instruction = [
            f'go to the {self.obj_color} {self.obj_type}',
            f'move to the {self.obj_color} {self.obj_type}',
            f'head toward the {self.obj_color} {self.obj_type}',
            f'walk to the {self.obj_color} {self.obj_type}',
            f'proceed to the {self.obj_color} {self.obj_type}',
            f'navigate to the {self.obj_color} {self.obj_type}',
            f'make your way to the {self.obj_color} {self.obj_type}',
            f'approach the {self.obj_color} {self.obj_type}',
            f'get to the {self.obj_color} {self.obj_type}',
            f'travel to the {self.obj_color} {self.obj_type}',
            f'head in the direction of the {self.obj_color} {self.obj_type}',
            f'reach the {self.obj_color} {self.obj_type}',
            f'step towards the {self.obj_color} {self.obj_type}',
            f'find your way to the {self.obj_color} {self.obj_type}',
            f'move in the direction of the {self.obj_color} {self.obj_type}',
            f'advance toward the {self.obj_color} {self.obj_type}',
            f'direct yourself to the {self.obj_color} {self.obj_type}',
            f'walk in the direction of the {self.obj_color} {self.obj_type}',
            f'proceed towards the {self.obj_color} {self.obj_type}',
            f'go over to the {self.obj_color} {self.obj_type}',
            f'head straight to the {self.obj_color} {self.obj_type}',
            f'locate and move to the {self.obj_color} {self.obj_type}',
            f'make a move towards the {self.obj_color} {self.obj_type}',
            f'position yourself near the {self.obj_color} {self.obj_type}',
            f'travel in the direction of the {self.obj_color} {self.obj_type}',
            f'get yourself to the {self.obj_color} {self.obj_type}',
            f'go in the direction of the {self.obj_color} {self.obj_type}',
            f'move toward the vicinity of the {self.obj_color} {self.obj_type}',
            f'walk over to the {self.obj_color} {self.obj_type}',
            f'take a step toward the {self.obj_color} {self.obj_type}',
            f'proceed in the direction of the {self.obj_color} {self.obj_type}',
            f'move closer to the {self.obj_color} {self.obj_type}',
            f'adjust your position to reach the {self.obj_color} {self.obj_type}',
            f'close the distance to the {self.obj_color} {self.obj_type}',
            f'advance in the direction of the {self.obj_color} {self.obj_type}',
            f'direct your steps toward the {self.obj_color} {self.obj_type}',
            f'find and approach the {self.obj_color} {self.obj_type}',
            f'shift your location to the {self.obj_color} {self.obj_type}',
            f'move in proximity to the {self.obj_color} {self.obj_type}',
            f'make your approach to the {self.obj_color} {self.obj_type}',
        ]

        self.novel_instruction = [
            f'proceed in the vicinity of the {self.obj_color} {self.obj_type}',
            f'move yourself toward the direction of the {self.obj_color} {self.obj_type}',
            f'head on over to the {self.obj_color} {self.obj_type}',
            f'make your path toward the {self.obj_color} {self.obj_type}',
            f'adjust your course to reach the {self.obj_color} {self.obj_type}',
            f'start moving toward the {self.obj_color} {self.obj_type}',
            f'set your direction toward the {self.obj_color} {self.obj_type}',
            f'walk in proximity to the {self.obj_color} {self.obj_type}',
            f'maneuver toward the {self.obj_color} {self.obj_type}',
            f'navigate yourself toward the {self.obj_color} {self.obj_type}',
        ]

    def gen_mission(self):
        self.place_agent(0, 0)
        if self.obj_type != 'door':
            obj, _ = self.add_object(0, 0, self.obj_type, self.obj_color)
        else:
            obj, _ = self.add_door(0, 0, color=self.obj_color, locked=False)
        self.add_distractors(0, 0, num_distractors=self.num_dists, all_unique=True)

        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
    
    def predict(self, state: np.ndarray = None) -> int:
        if state == None:
            state = self.my_encode()[0]
        return super().predict_base(state=state, task='goto', target_type=self.obj_type, target_color=self.obj_color)

    def dense_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        return super().dense_reward(state=state, next_state=next_state, target_type=self.obj_type, target_color=self.obj_color)

    def verify(self) -> bool:
        state = self.my_encode()[0]
        try:
            pos = get_target_pos(state=state, target_type=self.obj_type, target_color=self.obj_color)
        except:
            return False
        if np.array_equal(np.flip(pos), self.front_pos):
            return True
        else:
            return False


class OpenEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3, obj_color:str = COLOR_NAMES[0]):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.obj_color = obj_color
        self.obj_type = 'door'

        self.task_str = f'open the {self.obj_color} door'

        self.instruction = [
            f'open the {self.obj_color} door',
            f'please open the {self.obj_color} door',
            f'could you open the {self.obj_color} door?',
            f'unlock and open the {self.obj_color} door',
            f'push the {self.obj_color} door open',
            f'pull open the {self.obj_color} door',
            f'slide the {self.obj_color} door open',
            f'make sure the {self.obj_color} door is open',
            f'get the {self.obj_color} door open',
            f'let the {self.obj_color} door swing open',
            f'crack open the {self.obj_color} door',
            f'swing open the {self.obj_color} door',
            f'move the {self.obj_color} door so it is open',
            f'ensure the {self.obj_color} door is ajar',
            f'unlatch and open the {self.obj_color} door',
            f'hold the {self.obj_color} door open',
            f'pull back the {self.obj_color} door',
            f'push forward the {self.obj_color} door',
            f'give the {self.obj_color} door a push to open it',
            f'nudge the {self.obj_color} door open',
            f'tug the {self.obj_color} door open',
            f'open the door that has a {self.obj_color} hue',
            f'unfasten and open the {self.obj_color} door',
            f'please ensure the {self.obj_color} door is not closed',
            f'flip the latch and open the {self.obj_color} door',
            f'get the {self.obj_color} door to open',
            f'bring the {self.obj_color} door to an open position',
            f'please slide open the {self.obj_color} door',
            f'make sure the {self.obj_color} door is not shut',
            f'open up the {self.obj_color} door',
            f'disengage the lock and open the {self.obj_color} door',
            f'turn the handle and open the {self.obj_color} door',
            f'open the {self.obj_color} entrance',
            f'allow the {self.obj_color} door to be open',
            f'push or pull to open the {self.obj_color} door',
            f'let the {self.obj_color} door be open',
            f'don not keep the {self.obj_color} door closed, open it',
            f'adjust the {self.obj_color} door to be open',
            f'move the {self.obj_color} door out of the way',
            f'make the {self.obj_color} door accessible by opening it'
        ]

        self.novel_instruction = [
            f'leave the {self.obj_color} door open',
            f'push the {self.obj_color} door to open it fully',
            f'let the {self.obj_color} door remain open',
            f'move aside the {self.obj_color} door to open it',
            f'permit the {self.obj_color} door to stay ajar',
            f'manipulate the {self.obj_color} door into an open state',
            f'clear the way by opening the {self.obj_color} door',
            f'create an opening by unlocking the {self.obj_color} door',
            f'hold back the {self.obj_color} door so it stays open',
            f'ensure the passage by keeping the {self.obj_color} door unclosed'
        ]

    def gen_mission(self):
        self.place_agent(0, 0)
        obj, _ = self.add_door(0, 0, color=self.obj_color, locked=False)
        self.add_distractors(0, 0, num_distractors=self.num_dists, all_unique=True)

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(obj.type, obj.color))
    
    def predict(self, state: np.ndarray = None) -> int:
        if state == None:
            state = self.my_encode()[0]
        return super().predict_base(state=state, task='open', target_type='door', target_color=self.obj_color)
    
    def dense_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        return super().dense_reward(state=state, next_state=next_state, target_type='door', target_color=self.obj_color)

    def verify(self) -> bool:
        state = self.my_encode()[0]
        try:
            pos = get_target_pos(state=state, target_type='door', target_color=self.obj_color)
        except:
            return False
        if state[pos[0], pos[1], 2] == STATE_TO_IDX['open']:
            return True
        else:
            return False


class PickUpEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3, obj_type:str = OBJ_TYPES[0], obj_color:str = COLOR_NAMES[0]):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.obj_type = obj_type
        self.obj_color = obj_color

        self.task_str = f'pick up the {self.obj_color} {self.obj_type}'

        self.instruction = [
            f'pick up the {self.obj_color} {self.obj_type}',
            f'grab the {self.obj_color} {self.obj_type}',
            f'pick up the {self.obj_type} that is {self.obj_color}',
            f'retrieve the {self.obj_color} {self.obj_type}',
            f'lift the {self.obj_color} {self.obj_type}',
            f'take hold of the {self.obj_color} {self.obj_type}',
            f'get the {self.obj_color} {self.obj_type}',
            f'reach for the {self.obj_color} {self.obj_type}',
            f'secure the {self.obj_color} {self.obj_type}',
            f'take the {self.obj_color} {self.obj_type}',
            f'snag the {self.obj_color} {self.obj_type}',
            f'hold the {self.obj_color} {self.obj_type}',
            f'bring me the {self.obj_color} {self.obj_type}',
            f'collect the {self.obj_color} {self.obj_type}',
            f'obtain the {self.obj_color} {self.obj_type}',
            f'fetch the {self.obj_color} {self.obj_type}',
            f'scoop up the {self.obj_color} {self.obj_type}',
            f'snatch the {self.obj_color} {self.obj_type}',
            f'clutch the {self.obj_color} {self.obj_type}',
            f'grasp the {self.obj_color} {self.obj_type}',
            f'seize the {self.obj_color} {self.obj_type}',
            f'take possession of the {self.obj_color} {self.obj_type}',
            f'acquire the {self.obj_color} {self.obj_type}',
            f'hoist the {self.obj_color} {self.obj_type}',
            f'handle the {self.obj_color} {self.obj_type}',
            f'gather up the {self.obj_color} {self.obj_type}',
            f'yank the {self.obj_color} {self.obj_type}',
            f'lift up the {self.obj_color} {self.obj_type}',
            f'hold onto the {self.obj_color} {self.obj_type}',
            f'bring the {self.obj_color} {self.obj_type} to me',
            f'hand me the {self.obj_color} {self.obj_type}',
            f'carry the {self.obj_color} {self.obj_type}',
            f'take up the {self.obj_color} {self.obj_type}',
            f'retrieve and hold the {self.obj_color} {self.obj_type}',
            f'catch the {self.obj_color} {self.obj_type}',
            f'pull the {self.obj_color} {self.obj_type} toward you',
            f'move the {self.obj_color} {self.obj_type} into your hands',
            f'lift and hold the {self.obj_color} {self.obj_type}',
            f'get hold of the {self.obj_color} {self.obj_type}',
            f'reach out for the {self.obj_color} {self.obj_type}',
        ]

        self.novel_instruction = [
            f'grip the {self.obj_color} {self.obj_type}',
            f'snag hold of the {self.obj_color} {self.obj_type}',
            f'clasp the {self.obj_color} {self.obj_type}',
            f'reach over and take the {self.obj_color} {self.obj_type}',
            f'obtain and hold the {self.obj_color} {self.obj_type}',
            f'gather the {self.obj_color} {self.obj_type} into your hands',
            f'draw the {self.obj_color} {self.obj_type} toward yourself',
            f'take control of the {self.obj_color} {self.obj_type}',
            f'wrap your fingers around the {self.obj_color} {self.obj_type}',
            f'bring your hands to the {self.obj_color} {self.obj_type} and lift it',
        ]

    def gen_mission(self):
        self.place_agent(0, 0)
        obj, _ = self.add_object(0, 0, self.obj_type, self.obj_color)
        self.add_distractors(0, 0, num_distractors=self.num_dists, all_unique=True)

        self.check_objs_reachable()

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))
    
    def predict(self, state: np.ndarray = None) -> int:
        if state == None:
            state = self.my_encode()[0]
        return super().predict_base(state=state, task='pickup', target_type=self.obj_type, target_color=self.obj_color)

    def dense_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        return super().dense_reward(state=state, next_state=next_state, target_type=self.obj_type, target_color=self.obj_color)

    def verify(self) -> bool:
        if (self.carrying is not None) and (self.carrying.encode()[0] == OBJECT_TO_IDX[self.obj_type]) and (self.carrying.encode()[1] == COLOR_TO_IDX[self.obj_color]):
            return True
        else:
            return False


class PutNextEnv(MyGrid):
    def __init__(
            self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3, 
            obj1_type:str = OBJ_TYPES[0], obj1_color:str = COLOR_NAMES[0], 
            obj2_type:str = OBJ_TYPES[1], obj2_color:str = COLOR_NAMES[1]
            ):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.obj1_type = obj1_type
        self.obj1_color = obj1_color
        self.obj2_type = obj2_type
        self.obj2_color = obj2_color

        self.task_str = f'put the {self.obj1_color} {self.obj1_type} next to the {self.obj2_color} {self.obj2_type}'

        self.instruction = [
            f'put the {self.obj1_color} {self.obj1_type} next to the {self.obj2_color} {self.obj2_type}',
            f'place the {self.obj1_color} {self.obj1_type} beside the {self.obj2_color} {self.obj2_type}',
            f'move the {self.obj1_color} {self.obj1_type} close to the {self.obj2_color} {self.obj2_type}',
            f'set the {self.obj1_color} {self.obj1_type} adjacent to the {self.obj2_color} {self.obj2_type}',
            f'position the {self.obj1_color} {self.obj1_type} near the {self.obj2_color} {self.obj2_type}',
            f'arrange the {self.obj1_color} {self.obj1_type} alongside the {self.obj2_color} {self.obj2_type}',
            f'put the {self.obj1_color} {self.obj1_type} near the {self.obj2_color} {self.obj2_type}',
            f'align the {self.obj1_color} {self.obj1_type} next to the {self.obj2_color} {self.obj2_type}',
            f'lay the {self.obj1_color} {self.obj1_type} right beside the {self.obj2_color} {self.obj2_type}',
            f'place the {self.obj1_color} {self.obj1_type} next to the {self.obj2_color} {self.obj2_type} without gaps',
            f'keep the {self.obj1_color} {self.obj1_type} close to the {self.obj2_color} {self.obj2_type}',
            f'shift the {self.obj1_color} {self.obj1_type} to be next to the {self.obj2_color} {self.obj2_type}',
            f'bring the {self.obj1_color} {self.obj1_type} near the {self.obj2_color} {self.obj2_type}',
            f'make sure the {self.obj1_color} {self.obj1_type} is right next to the {self.obj2_color} {self.obj2_type}',
            f'place the {self.obj1_color} {self.obj1_type} in close proximity to the {self.obj2_color} {self.obj2_type}',
            f'position the {self.obj1_color} {self.obj1_type} adjacent to the {self.obj2_color} {self.obj2_type}',
            f'ensure the {self.obj1_color} {self.obj1_type} is beside the {self.obj2_color} {self.obj2_type}',
            f'put the {self.obj1_color} {self.obj1_type} directly next to the {self.obj2_color} {self.obj2_type}',
            f'arrange the {self.obj1_color} {self.obj1_type} so it sits next to the {self.obj2_color} {self.obj2_type}',
            f'move the {self.obj1_color} {self.obj1_type} and place it beside the {self.obj2_color} {self.obj2_type}',
            f'put the {self.obj1_color} {self.obj1_type} right next to the {self.obj2_color} {self.obj2_type}',
            f'align the {self.obj1_color} {self.obj1_type} next to the {self.obj2_color} {self.obj2_type} precisely',
            f'set the {self.obj1_color} {self.obj1_type} alongside the {self.obj2_color} {self.obj2_type} closely',
            f'lay down the {self.obj1_color} {self.obj1_type} beside the {self.obj2_color} {self.obj2_type}',
            f'place the {self.obj1_color} {self.obj1_type} next to the {self.obj2_color} {self.obj2_type} carefully',
            f'adjust the {self.obj1_color} {self.obj1_type} so that it is next to the {self.obj2_color} {self.obj2_type}',
            f'put the {self.obj1_color} {self.obj1_type} adjacent to the {self.obj2_color} {self.obj2_type} neatly',
            f'make sure the {self.obj1_color} {self.obj1_type} is in a position next to the {self.obj2_color} {self.obj2_type}',
            f'organize the {self.obj1_color} {self.obj1_type} right next to the {self.obj2_color} {self.obj2_type}',
            f'adjust the position of the {self.obj1_color} {self.obj1_type} to be next to the {self.obj2_color} {self.obj2_type}',
            f'have the {self.obj1_color} {self.obj1_type} sit near the {self.obj2_color} {self.obj2_type}',
            f'arrange the {self.obj1_color} {self.obj1_type} to be in close proximity to the {self.obj2_color} {self.obj2_type}',
            f'ensure that the {self.obj1_color} {self.obj1_type} is placed near the {self.obj2_color} {self.obj2_type}',
            f'shift the {self.obj1_color} {self.obj1_type} to be alongside the {self.obj2_color} {self.obj2_type}',
            f'keep the {self.obj1_color} {self.obj1_type} and {self.obj2_color} {self.obj2_type} side by side',
            f'make sure the {self.obj1_color} {self.obj1_type} is aligned next to the {self.obj2_color} {self.obj2_type}',
            f'put the {self.obj1_color} {self.obj1_type} in a position directly next to the {self.obj2_color} {self.obj2_type}',
            f'set the {self.obj1_color} {self.obj1_type} in close alignment with the {self.obj2_color} {self.obj2_type}',
            f'make the {self.obj1_color} {self.obj1_type} rest beside the {self.obj2_color} {self.obj2_type}',
            f'put the {self.obj1_color} {self.obj1_type} just next to the {self.obj2_color} {self.obj2_type}',
        ]

        self.novel_instruction = [
            f'position the {self.obj1_color} {self.obj1_type} right alongside the {self.obj2_color} {self.obj2_type}',
            f'ensure the {self.obj1_color} {self.obj1_type} is closely placed beside the {self.obj2_color} {self.obj2_type}',
            f'make the {self.obj1_color} {self.obj1_type} sit immediately next to the {self.obj2_color} {self.obj2_type}',
            f'arrange the {self.obj1_color} {self.obj1_type} neatly beside the {self.obj2_color} {self.obj2_type}',
            f'move the {self.obj1_color} {self.obj1_type} so that it is perfectly adjacent to the {self.obj2_color} {self.obj2_type}',
            f'keep the {self.obj1_color} {self.obj1_type} snugly next to the {self.obj2_color} {self.obj2_type}',
            f'lay the {self.obj1_color} {self.obj1_type} carefully next to the {self.obj2_color} {self.obj2_type}',
            f'adjust the {self.obj1_color} {self.obj1_type} to be in direct contact with the {self.obj2_color} {self.obj2_type}',
            f'place the {self.obj1_color} {self.obj1_type} so it touches the {self.obj2_color} {self.obj2_type}',
            f'ensure the {self.obj1_color} {self.obj1_type} is positioned seamlessly next to the {self.obj2_color} {self.obj2_type}',
        ]

        self.put_pos = None

    def gen_mission(self):
        self.place_agent(0, 0)
        obj1, _ = self.add_object(0, 0, self.obj1_type, self.obj1_color)
        obj2, _ = self.add_object(0, 0, self.obj2_type, self.obj2_color)
        self.add_distractors(0, 0, num_distractors=self.num_dists, all_unique=True)

        self.check_objs_reachable()

        self.instrs = PutNextInstr(ObjDesc(obj1.type, obj1.color), ObjDesc(obj2.type, obj2.color))
    
    def predict(self, state: np.ndarray = None) -> int:
        if state == None:
            state = self.my_encode()[0]
        target1_pos = get_target_pos(state=state, target_type=self.obj1_type, target_color=self.obj1_color)
        if not np.array_equal(target1_pos, get_agent_pos(state=state)):
            # 先pickup第一个object
            return super().predict_base(state=state, task='pickup', target_type=self.obj1_type, target_color=self.obj1_color)
        else:
            # 去第二个object旁边，再放下
            if self.put_pos is None:
                target_pos = get_target_pos(state=state, target_type=self.obj2_type, target_color=self.obj2_color)
                possible_diffs = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
                possible_targets = [target_pos+possible_diff for possible_diff in possible_diffs]
                for possible_target in possible_targets:
                    if (state[possible_target[0], possible_target[1], 0] == OBJECT_TO_IDX['empty']) and ((possible_target[1], possible_target[0]) in self.reachable):
                        self.put_pos = possible_target
                        break
                    elif np.array_equal(get_agent_pos(state=state), possible_target):
                        self.put_pos = possible_target
                        break
            distance = my_heuristic(target_pos=self.put_pos, state=state)
            if distance > 0:
                return super().predict_base(state=state, task='goto', target_position=self.put_pos)
            elif distance == 0:
                agent_pos = get_agent_pos(state=state)
                pos_diff = self.put_pos - agent_pos
                if np.array_equal(pos_diff, np.array([0, 1])):
                    need_dir = 'right'
                elif np.array_equal(pos_diff, np.array([0, -1])):
                    need_dir = 'left'
                elif np.array_equal(pos_diff, np.array([1, 0])):
                    need_dir = 'down'
                elif np.array_equal(pos_diff, np.array([-1, 0])):
                    need_dir = 'up'
                else:
                    raise NotImplementedError
                
                if need_dir == IDX_TO_DIR[state[agent_pos[0], agent_pos[1], 2]]:
                    self.put_pos = None
                    return 4
                else:
                    return 0
            else:
                return random.choice([0, 1, 2])
    
    def imagine_distance(self, state: np.ndarray, target_type, target_color) -> float:
        target1_pos = get_target_pos(state=state, target_type=self.obj1_type, target_color=self.obj1_color)
        target2_pos = get_target_pos(state=state, target_type=self.obj2_type, target_color=self.obj2_color)
        if not np.array_equal(target1_pos, get_agent_pos(state=state)):
            # 先pickup第一个object
            distance1 = my_heuristic(target_pos=target1_pos, state=state)
            distance2 = my_heuristic(target_pos=target2_pos, my_pos=target1_pos)
        else:
            # 第一个object已经pickup
            distance1 = 0
            distance2 = my_heuristic(target_pos=target2_pos, state=state)
        return distance1 + distance2
    
    def dense_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        return super().dense_reward(state=state, next_state=next_state, target_type=None, target_color=None)

    def verify(self) -> bool:
        state = self.my_encode()[0]
        try:
            pos1 = get_target_pos(state=state, target_type=self.obj1_type, target_color=self.obj1_color)
            pos2 = get_target_pos(state=state, target_type=self.obj2_type, target_color=self.obj2_color)
        except:
            return False
        if pos_next_to(pos_a=pos1, pos_b=pos2):
            return True
        else:
            return False


class GotoSeqEnv(MyGrid):
    def __init__(
            self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3, 
            obj1_type:str = OBJ_TYPES[0], obj1_color:str = COLOR_NAMES[0], 
            obj2_type:str = OBJ_TYPES[1], obj2_color:str = COLOR_NAMES[1]
            ):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.obj1_type = obj1_type
        self.obj1_color = obj1_color
        self.obj2_type = obj2_type
        self.obj2_color = obj2_color

        self.task_str = f'go to the {self.obj1_color} {self.obj1_type}, then goto the {self.obj2_color} {self.obj2_type}'

        self.instruction = [self.task_str for _ in range(10)]
    
    def gen_mission(self):
        self.place_agent(0, 0)
        if self.obj1_type != 'door':
            obj1, _ = self.add_object(0, 0, self.obj1_type, self.obj1_color)
        else:
            obj1, _ = self.add_door(0, 0, color=self.obj1_color, locked=False)
        if self.obj2_type != 'door':
            obj2, _ = self.add_object(0, 0, self.obj2_type, self.obj2_color)
        else:
            obj2, _ = self.add_door(0, 0, color=self.obj2_color, locked=False)
        self.add_distractors(0, 0, num_distractors=self.num_dists, all_unique=True)

        self.check_objs_reachable()

        instr_a = GoToInstr(ObjDesc(obj1.type, obj1.color))
        instr_b = GoToInstr(ObjDesc(obj2.type, obj2.color))
        self.instrs = BeforeInstr(instr_a, instr_b)
    
    def imagine_distance(self, state: np.ndarray, target_type:str, target_color:str) -> float:
        return 0.0

    def verify(self) -> bool:
        if not self.instr_a_done:
            state = self.my_encode()[0]
            try:
                pos = get_target_pos(state=state, target_type=self.obj1_type, target_color=self.obj1_color)
            except:
                return False
            if np.array_equal(np.flip(pos), self.front_pos):
                self.instr_a_done = True
                return self.verify()
            else:
                return False
        else:
            state = self.my_encode()[0]
            try:
                pos = get_target_pos(state=state, target_type=self.obj2_type, target_color=self.obj2_color)
            except:
                return False
            if np.array_equal(np.flip(pos), self.front_pos):
                return True
            else:
                return False


class PickUpSeqEnv(MyGrid):
    def __init__(
            self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3, 
            obj1_type:str = OBJ_TYPES[0], obj1_color:str = COLOR_NAMES[0], 
            obj2_type:str = OBJ_TYPES[1], obj2_color:str = COLOR_NAMES[1]
            ):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.obj1_type = obj1_type
        self.obj1_color = obj1_color
        self.obj2_type = obj2_type
        self.obj2_color = obj2_color

        self.task_str = f'pick up the {self.obj1_color} {self.obj1_type}, then put it down, then pick up the {self.obj2_color} {self.obj2_type}'

        self.instruction = [self.task_str for _ in range(10)]
    
    def gen_mission(self):
        self.place_agent(0, 0)
        obj1, _ = self.add_object(0, 0, self.obj1_type, self.obj1_color)
        obj2, _ = self.add_object(0, 0, self.obj2_type, self.obj2_color)
        self.add_distractors(0, 0, num_distractors=self.num_dists, all_unique=True)

        self.check_objs_reachable()

        instr_a = PickupInstr(ObjDesc(obj1.type, obj1.color))
        instr_b = PickupInstr(ObjDesc(obj2.type, obj2.color))
        self.instrs = BeforeInstr(instr_a, instr_b)
    
    def imagine_distance(self, state: np.ndarray, target_type:str, target_color:str) -> float:
        return 0.0
    
    def verify(self) -> bool:
        if not self.instr_a_done:
            if (self.carrying is not None) and (self.carrying.encode()[0] == OBJECT_TO_IDX[self.obj1_type]) and (self.carrying.encode()[1] == COLOR_TO_IDX[self.obj1_color]):
                self.instr_a_done = True
                return self.verify()
            else:
                return False
        else:
            if (self.carrying is not None) and (self.carrying.encode()[0] == OBJECT_TO_IDX[self.obj2_type]) and (self.carrying.encode()[1] == COLOR_TO_IDX[self.obj2_color]):
                return True
            else:
                return False


class GoSEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.task_str = f'go straight'

        self.start_pos: Tuple[int, int]
        self.start_dir: int

        self.instruction = [self.task_str for _ in range(10)]

    def gen_mission(self):
        self.place_agent(0, 0)

        obj, _ = self.add_door(0, 0, color='red', locked=False)

        self.add_distractors(0, 0, num_distractors=self.num_dists, all_unique=True)
        
        while isinstance(self.unwrapped.grid.get(self.front_pos[0], self.front_pos[1]), Wall) or isinstance(self.unwrapped.grid.get(self.front_pos[0], self.front_pos[1]), Door):
            self.agent_dir += 1
            if self.agent_dir >= 4:
                self.agent_dir -= 4
        self.unwrapped.grid.set(self.front_pos[0], self.front_pos[1], None)

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(obj.type, obj.color))  # 仅做占位用
    
    def imagine_distance(self, state: np.ndarray, target_type:str, target_color:str) -> float:
        return 0.0
    
    def verify(self):
        current_pos = (self.agent_pos[1], self.agent_pos[0])
        if self.start_dir == 0:  # right
            if current_pos[0] == self.start_pos[0]:  # 行不变
                if current_pos[1] > self.start_pos[1]:  # 列变大
                    return True
        elif self.start_dir == 1:  # down
            if current_pos[1] == self.start_pos[1]:  # 列不变
                if current_pos[0] > self.start_pos[0]:  # 行变大
                    return True
        elif self.start_dir == 2:  # left
            if current_pos[0] == self.start_pos[0]:  # 行不变
                if current_pos[1] < self.start_pos[1]:  # 列变小
                    return True
        elif self.start_dir == 3:  # up
            if current_pos[1] == self.start_pos[1]:  # 列不变
                if current_pos[0] < self.start_pos[0]:  # 行变小
                    return True
        else:
            raise NotImplementedError
        
        return False


class GoTEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)   
        self.task_str = f'turn left or right, then go straight'

        self.start_pos: Tuple[int, int]
        self.start_dir: int

        self.instruction = [self.task_str for _ in range(10)]

    def gen_mission(self):
        self.place_agent(0, 0)

        obj, _ = self.add_door(0, 0, color='red', locked=False)

        self.add_distractors(0, 0, num_distractors=self.num_dists, all_unique=True)

        if not isinstance(self.unwrapped.grid.get(self.agent_pos[0]-1, self.agent_pos[1]), Wall) and (not isinstance(self.unwrapped.grid.get(self.agent_pos[0]-1, self.agent_pos[1]), Door)):
            self.unwrapped.grid.set(self.agent_pos[0]-1, self.agent_pos[1], None)
        if not isinstance(self.unwrapped.grid.get(self.agent_pos[0]+1, self.agent_pos[1]), Wall) and (not isinstance(self.unwrapped.grid.get(self.agent_pos[0]-1, self.agent_pos[1]), Door)):
            self.unwrapped.grid.set(self.agent_pos[0]-1, self.agent_pos[1], None)

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(obj.type, obj.color))  # 仅做占位用
    
    def imagine_distance(self, state: np.ndarray, target_type:str, target_color:str) -> float:
        return 0.0

    def verify(self):
        current_pos = (self.agent_pos[1], self.agent_pos[0])
        if self.start_dir in [0, 2]:  # right, left
            if current_pos[1] == self.start_pos[1]:  # 列不变
                if current_pos[0] != self.start_pos[0]:  # 行变
                    return True
        elif self.start_dir in [1, 3]:  # down, up
            if current_pos[0] == self.start_pos[0]:  # 行不变
                if current_pos[1] != self.start_pos[1]:  # 列变
                    return True
        else:
            raise NotImplementedError
        
        return False


class PutLineEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 0, obj_color:str = COLOR_NAMES[0]):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.obj1_type = 'ball'
        self.obj1_color = obj_color
        self.obj2_type = 'box'
        self.obj2_color = obj_color
        self.obj3_type = 'key'
        self.obj3_color = obj_color
        
        self.task_str = f'put the three {obj_color} items in a line'

        self.instruction = [self.task_str for _ in range(10)]

    def gen_mission(self):
        self.place_agent(0, 0)

        obj1, _ = self.add_object(0, 0, self.obj1_type, self.obj1_color)
        obj2, _ = self.add_object(0, 0, self.obj2_type, self.obj2_color)
        obj3, _ = self.add_object(0, 0, self.obj3_type, self.obj3_color)

        obj, _ = self.add_door(0, 0, color='red', locked=False)

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(obj.type, obj.color))  # 仅做占位用
    
    def imagine_distance(self, state: np.ndarray, target_type:str, target_color:str) -> float:
        return 0.0

    def verify(self):
        state = self.my_encode()[0]
        try:
            pos1 = get_target_pos(state=state, target_type=self.obj1_type, target_color=self.obj1_color)
            pos2 = get_target_pos(state=state, target_type=self.obj2_type, target_color=self.obj2_color)
            pos3 = get_target_pos(state=state, target_type=self.obj3_type, target_color=self.obj3_color)
        except:
            return False
        if (pos1[0] == pos2[0] == pos3[0]) or (pos1[1] == pos2[1] == pos3[1]):
            return True
        else:
            return False


class PutPileEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 0, obj_color:str = COLOR_NAMES[0]):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.obj1_type = 'ball'
        self.obj1_color = obj_color
        self.obj2_type = 'box'
        self.obj2_color = obj_color
        self.obj3_type = 'key'
        self.obj3_color = obj_color

        self.obj_type = None
        self.obj_color = None
        
        self.task_str = f'gather the three {obj_color} items into a pile'

        self.instruction = [self.task_str for _ in range(10)]

    def gen_mission(self):
        self.place_agent(0, 0)

        obj1, _ = self.add_object(0, 0, self.obj1_type, self.obj1_color)
        obj2, _ = self.add_object(0, 0, self.obj2_type, self.obj2_color)
        obj3, _ = self.add_object(0, 0, self.obj3_type, self.obj3_color)

        obj, _ = self.add_door(0, 0, color='red', locked=False)

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(obj.type, obj.color))  # 仅做占位用
    
    def imagine_distance(self, state: np.ndarray, target_type:str, target_color:str) -> float:
        return 0.0

    def verify(self):
        state = self.my_encode()[0]
        try:
            pos1 = get_target_pos(state=state, target_type=self.obj1_type, target_color=self.obj1_color)
            pos2 = get_target_pos(state=state, target_type=self.obj2_type, target_color=self.obj2_color)
            pos3 = get_target_pos(state=state, target_type=self.obj3_type, target_color=self.obj3_color)
        except:
            return False

        pile_count = 0
        if pos_next_to(pos_a=pos1, pos_b=pos2):
            pile_count += 1
        if pos_next_to(pos_a=pos1, pos_b=pos3):
            pile_count += 1
        if pos_next_to(pos_a=pos2, pos_b=pos2):
            pile_count += 1
        
        if pile_count >= 2:
            return True
        else:
            return False


class OpenLockEnv(MyGrid):
    def __init__(
            self, room_size:int = 8, max_steps:int | None = None, num_dists:int = 3, 
            obj1_color:str = COLOR_NAMES[0], obj2_color:str = COLOR_NAMES[1]
            ):
        super().__init__(room_size=room_size, max_steps=max_steps, num_dists=num_dists)
        self.obj1_type = 'key'
        self.obj1_color = obj1_color
        self.obj2_type = 'door'
        self.obj2_color = obj2_color

        self.task_str = f'pick up the {self.obj1_color} key, then open the {self.obj2_color} door'

        self.instruction = [self.task_str for _ in range(10)]

    def gen_mission(self):
        self.place_agent(0, 0)
        obj1, _ = self.add_object(0, 0, self.obj1_type, self.obj1_color)
        obj2, _ = self.add_door(0, 0, color=self.obj_color, locked=False)
        self.add_distractors(0, 0, num_distractors=self.num_dists, all_unique=True)

        self.check_objs_reachable()

        instr_a = PickupInstr(ObjDesc(obj1.type, obj1.color))
        instr_b = OpenInstr(ObjDesc(obj2.type, obj2.color))
        self.instrs = BeforeInstr(instr_a, instr_b)

    def verify(self) -> bool:
        state = self.my_encode()[0]
        try:
            pos = get_target_pos(state=state, target_type='door', target_color=self.obj2_color)
        except:
            return False
        if state[pos[0], pos[1], 2] == STATE_TO_IDX['open']:
            return True
        else:
            return False
    
    def imagine_distance(self, state: np.ndarray, target_type:str, target_color:str) -> float:
        return 0.0


class DenseRewardWrapper(Wrapper):
    def __init__(self, env: MyGrid):
        Wrapper.__init__(self, env)
        self.env: MyGrid

    def step(self, action):
        state = self.env.my_encode()[0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward < 0.1:
            reward += self.env.dense_reward(state=state, next_state=self.env.my_encode()[0])
        return obs, reward, terminated, truncated, info


class EncodeWrapper(Wrapper):
    def __init__(self, env: MyGrid):
        Wrapper.__init__(self, env)
        assert isinstance(self.env.unwrapped, MyGrid)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(((self.env.unwrapped.room_size-1)*(self.env.unwrapped.room_size-2)+1)*3, ),
            dtype="uint8",
        )

        self.observation_space = new_image_space

    def observation(self, obs):
        assert isinstance(self.env.unwrapped, MyGrid)

        return self.env.unwrapped.my_state()[0]
    
    def step(self, action):
        assert isinstance(self.env.unwrapped, MyGrid)

        obs, reward, terminated, truncated, info = self.env.step(action)
        info['view_encode'] = obs['image'] if isinstance(obs, dict) else obs
        return self.env.unwrapped.my_state()[0], reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        assert isinstance(self.env.unwrapped, MyGrid)
        
        obs, info = self.env.reset(**kwargs)
        info['view_encode'] = obs['image'] if isinstance(obs, dict) else obs
        return self.env.unwrapped.my_state()[0], info


class MyWrapper(Wrapper):
    def __init__(self, env: MyGrid):
        Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        assert isinstance(self.env.unwrapped, MyGrid)
        self.env.unwrapped.future_actions = None
        self.env.unwrapped.instr_a_done = False
        obs, info = self.env.reset(**kwargs)
        self.env.unwrapped.distance0 = self.env.unwrapped.imagine_distance(state=self.env.unwrapped.my_encode()[0], target_type=self.env.unwrapped.obj_type, target_color=self.env.unwrapped.obj_color)
        return obs, info
