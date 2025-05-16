import numpy as np

from minigrid.envs.babyai.core.verifier import pos_next_to
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from base import MyGrid
from grid_utils import OBJ_TYPES
from grid_utils import my_heuristic, get_agent_pos, get_target_pos


class GoToEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, obj_color:str = COLOR_NAMES[0], obj_type:str = OBJ_TYPES[0]):
        super().__init__(room_size=room_size, max_steps=max_steps)
        self.obj_color = obj_color
        self.obj_type = obj_type
        self.obj_type_color_dict[self.obj_type] = self.obj_color
        
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

        self.env_name = 'goto'

    def predict(self) -> int:
        state = self.my_state()
        target_pos = get_target_pos(state=state, target_color=self.obj_color, target_type=self.obj_type)
        if self.obj_type == 'door':
            target_pos += np.array([0, -1])  # need to reach the left of the door
        agent_pos = get_agent_pos(state=state)
        return self.predict_goto(agent_pos=agent_pos, target_pos=target_pos)

    def verify(self) -> bool:
        state = self.my_state()
        target_pos = get_target_pos(state=state, target_color=self.obj_color, target_type=self.obj_type)
        if target_pos is None:
            return False  # can not find target
        
        if self.obj_type == 'door':
            target_pos += np.array([0, -1])  # need to go to the left of the door
        
        if np.array_equal(np.flip(target_pos), self.agent_pos):
            return True
        else:
            return False


class OpenEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, obj_color:str = COLOR_NAMES[0]):
        super().__init__(room_size=room_size, max_steps=max_steps)
        self.obj_color = obj_color
        self.obj_type = 'door'
        self.obj_type_color_dict[self.obj_type] = self.obj_color

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

        self.env_name = 'open'

    def predict(self) -> int:
        state = self.my_state()
        target_pos = get_target_pos(state=state, target_color=self.obj_color, target_type=self.obj_type)
        if self.obj_type == 'door':
            target_pos += np.array([0, -1])  # need to reach the left of the door
        agent_pos = get_agent_pos(state=state)
        if not np.array_equal(agent_pos, target_pos):
            return self.predict_goto(agent_pos=agent_pos, target_pos=target_pos)
        else:
            return self.actions.toggle

    def verify(self) -> bool:
        state = self.my_state()
        target_pos = get_target_pos(state=state, target_type='door', target_color=self.obj_color)
        if target_pos is None:
            return False  # can not find target
        
        if state[target_pos[0], target_pos[1], 2] == STATE_TO_IDX['open']:
            return True
        else:
            return False


class PickUpEnv(MyGrid):
    def __init__(self, room_size:int = 8, max_steps:int | None = None, obj_color:str = COLOR_NAMES[0], obj_type:str = OBJ_TYPES[0]):
        super().__init__(room_size=room_size, max_steps=max_steps)
        self.obj_color = obj_color
        self.obj_type = obj_type
        self.obj_type_color_dict[self.obj_type] = self.obj_color

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
    
        self.env_name = 'pickup'
    
    def predict(self) -> int:
        state = self.my_state()
        target_pos = get_target_pos(state=state, target_color=self.obj_color, target_type=self.obj_type)
        agent_pos = get_agent_pos(state=state)
        if not np.array_equal(agent_pos, target_pos):
            return self.predict_goto(agent_pos=agent_pos, target_pos=target_pos)
        else:
            return self.actions.pickup

    def verify(self) -> bool:
        if (self.carrying is not None) and (self.carrying.type == self.obj_type) and (self.carrying.color == self.obj_color):
            return True
        else:
            return False


class PutNextEnv(MyGrid):
    def __init__(
            self, room_size:int = 8, max_steps:int | None = None,
            obj1_color:str = COLOR_NAMES[0], obj1_type:str = OBJ_TYPES[0],
            obj2_color:str = COLOR_NAMES[1], obj2_type:str = OBJ_TYPES[1],
            ):
        super().__init__(room_size=room_size, max_steps=max_steps)
        self.obj1_color = obj1_color
        self.obj1_type = obj1_type
        self.obj2_color = obj2_color
        self.obj2_type = obj2_type
        self.obj_type_color_dict[self.obj1_type] = self.obj1_color
        self.obj_type_color_dict[self.obj2_type] = self.obj2_color

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
        self.env_name = 'putnext'

    def gen_mission(self):
        super().gen_mission()
        self.check_objs_reachable()
    
    def predict(self) -> int:
        state = self.my_state()
        if self.carrying is None:
            # pick up the first object
            target1_pos = get_target_pos(state=state, target_color=self.obj1_color, target_type=self.obj1_type)
            agent_pos = get_agent_pos(state=state)
            if not np.array_equal(agent_pos, target1_pos):
                return self.predict_goto(agent_pos=agent_pos, target_pos=target1_pos)
            else:
                return self.actions.pickup
        else:
            # find position to put down the first object
            if self.put_pos is None:
                target2_pos = get_target_pos(state=state, target_color=self.obj2_color, target_type=self.obj2_type)
                possible_diff_list = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
                possible_targets = [target2_pos+possible_diff for possible_diff in possible_diff_list]
                for possible_target in possible_targets:
                    if state[possible_target[0], possible_target[1], 0] == OBJECT_TO_IDX['empty']:
                        self.put_pos = possible_target
                        break

            # go next to the second object, then drop
            agent_pos = get_agent_pos(state=state)
            if not np.array_equal(agent_pos, self.put_pos):
                return self.predict_goto(agent_pos=agent_pos, target_pos=self.put_pos)
            else:
                self.put_pos = None
                return self.actions.drop

    def verify(self) -> bool:
        state = self.my_state()
        pos1 = get_target_pos(state=state, target_type=self.obj1_type, target_color=self.obj1_color)
        pos2 = get_target_pos(state=state, target_type=self.obj2_type, target_color=self.obj2_color)
        if (pos1 is None) or (pos2 is None):
            return False  # can not find target
        
        if pos_next_to(pos_a=pos1, pos_b=pos2):
            return True
        else:
            return False

    def estimated_distance(self, state: np.ndarray) -> float:
        target1_pos = get_target_pos(state=state, target_color=self.obj1_color, target_type=self.obj1_type)
        target2_pos = get_target_pos(state=state, target_color=self.obj2_color, target_type=self.obj2_type)
        if target1_pos is None:
            target1_pos = self.obj_last_pos_dict[self.obj1_type]
            assert target1_pos is not None
        if target2_pos is None:
            target2_pos = self.obj_last_pos_dict[self.obj2_type]
            assert target2_pos is not None
        
        if state[0, 0, 0] == OBJECT_TO_IDX['unseen']:
            # no pick up yet
            distance1 = my_heuristic(target_pos=target1_pos, state=state)
            distance2 = my_heuristic(target_pos=target2_pos, state=state, start_pos=target1_pos)
            penalty = self.base_penalty
        elif not ((state[0, 0, 0] == OBJECT_TO_IDX[self.obj1_type]) and (state[0, 0, 1] == COLOR_TO_IDX[self.obj1_color])):
            # pick up wrong object
            distance1 = 0.0
            distance2 = 0.0
            penalty = self.base_penalty * 3
        else:
            # first object already picked up
            distance1 = 0.0
            distance2 = my_heuristic(target_pos=target2_pos, state=state)
            penalty = 0.0
        return distance1 + distance2 + penalty
