from typing import Union

import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX


OBJ_TYPES = ['ball', 'box', 'key', 'door']

DIRS = ['right', 'down', 'left', 'up']
IDX_TO_DIR = {}
for DIR in DIRS:
    IDX_TO_DIR[OBJECT_TO_IDX['agent'] * (DIRS.index(DIR)+1)] = DIR

CUSTOM_ACTION_TO_NEED_DIR = {
    Actions.left: 2,  # 0, go left
    Actions.right: 0,  # 1, go right
    Actions.forward: 3,  # 2, go up
    Actions.done: 1,  # 6, go down
}


def get_agent_pos(state: np.ndarray) -> Union[np.ndarray, None]:
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i, j, 2] in IDX_TO_DIR.keys():
                return np.array([i, j])
    return None


def get_target_pos(state: np.ndarray, target_color: int, target_type: str) -> Union[np.ndarray, None]:
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i, j, 0] == OBJECT_TO_IDX[target_type] and state[i, j, 1] == COLOR_TO_IDX[target_color]:
                return np.array([i, j])
    return None


def my_heuristic(target_pos: np.ndarray, state: np.ndarray, start_pos: np.ndarray = None) -> float:
    if start_pos is None:
        start_pos = get_agent_pos(state=state)
    if not (target_pos == 0).all():
        distance = np.linalg.norm(x=start_pos-target_pos, ord=1, keepdims=False)
    else:
        target_pos = get_agent_pos(state=state)
        distance = np.linalg.norm(x=start_pos-target_pos, ord=1, keepdims=False) + 1
    return float(distance)
