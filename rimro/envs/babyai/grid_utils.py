from copy import deepcopy
from typing import List, Union

import heapq
import numpy as np
from tqdm import tqdm
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX


baseline_env_name_list = ['goto', 'open', 'pickup', 'putnext']
rephrase_level_env_name_list = baseline_env_name_list
easy_level_env_name_list = ['goto_seq', 'pickup_seq', 'go_straight', 'go_turn']
hard_level_env_name_list = ['open_lock', 'put_line', 'put_pile']

OBJ_TYPES = ['ball', 'box', 'key', 'door']

DIRS = ['right', 'down', 'left', 'up']
IDX_TO_DIR = {}
for DIR in DIRS:
    IDX_TO_DIR[OBJECT_TO_IDX['agent'] * (DIRS.index(DIR)+1)] = DIR


def my_heuristic(target_pos: np.ndarray, state: np.ndarray = None, my_pos: np.ndarray = None) -> float:
    if (state is not None) and (my_pos is None):
        agent_pos = get_agent_pos(state=state)
    elif (state is None) and (my_pos is not None):
        agent_pos = my_pos
    else:
        raise NotImplementedError
    distance = np.linalg.norm(x=agent_pos-target_pos, ord=1, keepdims=False)
    return float(distance - 1)


def get_agent_pos(state: np.ndarray) -> np.ndarray:
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i, j, 2] in IDX_TO_DIR.keys():
                return np.array([i, j])
    raise NotImplementedError


class State(object):
    def __init__(self, state: np.ndarray, target_pos: np.ndarray = None, father = None, action:int = None):
        self.state = state
        self.heuristic = my_heuristic

        if (target_pos is not None) and (father is None) and (action is None):
            self.target_pos = target_pos
            self.cost = 0
            self.action_history = []
        elif (target_pos is None) and (father is not None) and (action is not None):
            assert isinstance(father, State)
            self.target_pos = father.target_pos
            self.cost = father.cost + 1
            self.action_history = deepcopy(father.action_history)
            self.action_history.append(action)
        else:
            raise NotImplementedError
    
    def isGoalState(self) -> bool:
        return int(my_heuristic(target_pos=self.target_pos, state=self.state)) == 0
    
    def getSuccessors(self) -> List:
        agent_pos = get_agent_pos(state=self.state)
        agent_dir_idx = self.state[agent_pos[0], agent_pos[1], 2]
        agent_dir_idx_raw = agent_dir_idx // OBJECT_TO_IDX['agent'] - 1
        agent_dir = IDX_TO_DIR[agent_dir_idx]

        successors = []
        for action in [0, 1]:
            new_state = self.state.copy()
            if action == 0:
                new_agent_dir_raw = (agent_dir_idx_raw - 1) % 4
            elif action == 1:
                new_agent_dir_raw = (agent_dir_idx_raw + 1) % 4
            else:
                raise NotImplementedError
            new_state[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX['agent'] * (new_agent_dir_raw + 1)
            successors.append(State(state=new_state, father=self, action=action))

        if agent_dir == 'right':
            front_change = np.array([0, 1])
        elif agent_dir == 'down':
            front_change = np.array([1, 0])
        elif agent_dir == 'left':
            front_change = np.array([0, -1])
        elif agent_dir == 'up':
            front_change = np.array([-1, 0])
        else:
            raise NotImplementedError
        front_pos = agent_pos + front_change
        if self.state[front_pos[0], front_pos[1], 0] == OBJECT_TO_IDX['empty']:
            new_state = self.state.copy()
            new_state[agent_pos[0], agent_pos[1]] = self.state[front_pos[0], front_pos[1]]
            new_state[front_pos[0], front_pos[1]] = self.state[agent_pos[0], agent_pos[1]]
            successors.append(State(state=new_state, father=self, action=2))

        return successors

    def get_imaginary_cost(self) -> float:
        return float(self.cost) + self.heuristic(target_pos=self.target_pos, state=self.state)

    def __lt__(self, other) -> bool:
        assert isinstance(other, State)
        return self.get_imaginary_cost() < other.get_imaginary_cost()
    
    def __eq__(self, other) -> bool:
        assert isinstance(other, State)
        return np.array_equal(self.state, other.state)
            
    def __hash__(self) -> int:
        return hash(self.state.tostring())


class StateContainer(object):
    def __init__(self, start_state: State = None) -> None:
        self.container = [start_state]
        heapq.heapify(self.container)

    def push(self, state: State) -> None:
        heapq.heappush(self.container, state)
    
    def pop(self) -> State:
        return heapq.heappop(self.container)


def find_path_to(target_pos: np.ndarray, state: np.ndarray) -> Union[list[int], None]:
    visited_states = set()
    visited_state_num = -1

    start_state = State(state=state, target_pos=target_pos)
    container = StateContainer(start_state=start_state)
    container_len = 1

    pbar = tqdm(desc='searched_states', leave=False, ncols=120, disable=True)
    while container_len > 0:
        current_state = container.pop()
        container_len -= 1

        visited_states.add(current_state)
        visited_state_num += 1

        pbar.update()
        pbar.set_postfix({'cost': current_state.cost, 'visit_num': visited_state_num, 'container_len': container_len})

        if current_state.isGoalState():
            return current_state.action_history
        else:
            successors = current_state.getSuccessors()
            for successor in successors:
                assert isinstance(successor, State)
                if successor not in visited_states:
                    container.push(successor)
                    container_len += 1

    return None


def get_target_pos(state: np.ndarray, target_type: str, target_color: int) -> np.ndarray:
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i, j, 0] == OBJECT_TO_IDX[target_type] and state[i, j, 1] == COLOR_TO_IDX[target_color]:
                return np.array([i, j])
    raise NotImplementedError
