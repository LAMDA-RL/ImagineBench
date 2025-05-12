from typing import List, Dict, Any, Optional
from itertools import chain

import numpy as np

CLEVR_QPOS_OBS_INDICES = lambda num: list(chain(*[[i*7, i*7+1] for i in range(num)]))

GOAL_SRC = 1
goal_src = 1
GOAL_DST = 2
goal_dst = 2
color_list = ['red', 'blue', 'green', 'purple', 'cyan']
MIN_PUNISH_DIST = 0.1
FAIL_DIST_THRESHOLD = 0.2
SUCC_DIST_THRESHOLD = 0.13 * 2 + 0.13 * 1
SINGLE_STEP_MOVE_DIST = 0.15  # define by ClevrEnv

EPS = 1e-3

BEHIND = 0  # +y
LEFT = 1  # -x
FRONT = 2  # -y
RIGHT = 3  # +x
STILL = 4
DIRECTIONS = [
    BEHIND,
    LEFT,
    FRONT,
    RIGHT,
]
DIRECTIONS_CNT = len(DIRECTIONS)
STEP_SIZE = 360 / DIRECTIONS_CNT
RG_SIZE = STEP_SIZE / 2
DIRECTION_TO_ANGLE_RG = {
    0: np.array([90 - RG_SIZE, 90 + RG_SIZE]),
}
for dir in range(1, DIRECTIONS_CNT):
    DIRECTION_TO_ANGLE_RG[dir] = (DIRECTION_TO_ANGLE_RG[dir - 1] + STEP_SIZE) % 360

# step-level utils
SELECT_MARK = 1
ACTION_DIRECTIONS = [
    [1, 0],       # 0
    [0, 1],       # 1
    [-1, 0],      # 2
    [0, -1],      # 3
    [0.8, 0.8],   # 4
    [-0.8, 0.8],  # 5
    [0.8, -0.8],  # 6
    [-0.8, -0.8], # 7
]
dir2action_list = {
    BEHIND: [1, 4, 5],
    LEFT:   [2, 5, 7],
    FRONT:  [3, 6, 7],
    RIGHT:  [0, 4, 6],
}


# Indicate for task-level terminal fn
TAU_2=0
TAU_3=1
ORDER=2
SORT=3


def _goal_xy(num_obj: int, obs: np.ndarray, goals: np.ndarray, label: int):
    obs = obs[:, :2*num_obj].reshape((obs.shape[0], num_obj, -1))
    goal_idx = np.where(goals[:, :num_obj] == label)

    return obs[goal_idx[0], goal_idx[1], :].copy()


def _compute_dist(num_obj: int, obs: np.ndarray, goals: np.ndarray):
    goal_src_xy = _goal_xy(num_obj, obs, goals, GOAL_SRC)
    goal_dst_xy = _goal_xy(num_obj, obs, goals, GOAL_DST)

    return np.linalg.norm(goal_src_xy - goal_dst_xy, axis=-1)


def _compute_angle(num_obj: int, obs: np.ndarray, goal: np.ndarray):
    goal_src_xy = _goal_xy(num_obj, obs, goal, GOAL_SRC)
    goal_dst_xy = _goal_xy(num_obj, obs, goal, GOAL_DST)
    delta_xy = goal_src_xy - goal_dst_xy
    delta_xy = np.where(np.abs(delta_xy) >= EPS, delta_xy, 0)

    cosine = delta_xy[:, 0] / (np.linalg.norm(delta_xy, axis=-1) + 1e-6)
    # cosine = delta_xy[:, 0] / np.linalg.norm(delta_xy, axis=-1)
    negative_flag = delta_xy[:, 1] < 0
    angles = np.arccos(cosine)
    angles[negative_flag] = 2 * np.pi - angles[negative_flag]

    return angles * 180 / np.pi


def _compute_bias(num_obj: int, obs: np.ndarray, init_obs: np.ndarray, goals: np.ndarray):
    init_obs = init_obs[:, :2*num_obj].reshape((init_obs.shape[0], num_obj, -1))
    obs = obs[:, :2*num_obj].reshape((obs.shape[0], num_obj, -1))

    goal_src_idx = np.where(goals[:, :num_obj] == GOAL_SRC)

    bias_arr = obs - init_obs
    bias_arr = bias_arr[goal_src_idx[0], goal_src_idx[1], :]
    bias = np.linalg.norm(bias_arr, axis=-1)

    return np.where(bias >= MIN_PUNISH_DIST, bias, 0)


def _success(num_obj: int, obs: np.ndarray, goals: np.ndarray):
    # angle check
    dirs = goals[:, -1]
    angles = _compute_angle(num_obj, obs, goals)
    angle_rgs = np.array([DIRECTION_TO_ANGLE_RG[int(d)] for d in dirs])
    angle_flags = np.zeros_like(dirs, dtype=bool)
    last_flag = angle_rgs[:, 1] < angle_rgs[:, 0]
    angle_flags[last_flag] = np.logical_or(
        np.logical_and(angle_rgs[:, 0] <= angles, angles < 360),
        np.logical_and(0 <= angles, angles < angle_rgs[:, 1]),
    )[last_flag]
    angle_flags[~last_flag] = np.logical_and(
        angle_rgs[:, 0] <= angles,
        angles < angle_rgs[:, 1],
    )[~last_flag]

    # distance check
    dist = _compute_dist(num_obj, obs, goals)
    dist_flags = dist < SUCC_DIST_THRESHOLD + EPS

    return np.logical_and(angle_flags, dist_flags)


def _failure(num_obj: int, obs: np.ndarray, init_obs: np.ndarray, goals: np.ndarray):
    """NOTE: Deprecated"""
    bias = _compute_bias(num_obj, obs, init_obs, goals)

    return np.any(bias > FAIL_DIST_THRESHOLD)


def step_level_a_judge(num_obj, observations, hist_observations, actions, goals) -> dict:
    batch_size = goals.shape[0]
    done_list = []
    succ_list = []
    still_dist_threshold = 0.05
    done = True
    for data_idx in range(batch_size):
        goal = goals[data_idx]
        goal_indicate = int(goal[num_obj])
        action = int(actions[data_idx])
        prev_obs = hist_observations[data_idx, 0]
        curr_obs = hist_observations[data_idx, 1]
        obs_diff = curr_obs - prev_obs
        if goal_indicate == STILL:
            succ = (obs_diff.abs().max() < still_dist_threshold).item()
        else:
            target_obj_tuple = np.where(goal[:num_obj] == SELECT_MARK)
            assert len(target_obj_tuple) == 1
            target_obj = target_obj_tuple[0].item()
            obj_select = action // len(ACTION_DIRECTIONS)
            dir_select = action % len(ACTION_DIRECTIONS)
            succ = obj_select == target_obj and dir_select in dir2action_list[goal_indicate]
        done_list.append(done)
        succ_list.append(succ)

    done = np.array(done_list)
    succ = np.array(succ_list)

    judge_result = dict(
        done=done,
        succ=succ,
    )

    return judge_result


def step_level_s_judge(num_obj, observations, hist_observations, actions, goals) -> dict:
    assert hist_observations.shape[1] == 2, 'Step level support 2 trainsition only!'

    batch_size = goals.shape[0]
    done_list = []
    succ_list = []
    still_dist_threshold = 0.05
    x_axis_idx_arr = np.array([0, 2, 4, 6, 8])
    y_axis_idx_arr = np.array([1, 3, 5, 7, 9])
    done = True
    for data_idx in range(batch_size):
        goal_indicate = goals[data_idx, num_obj]
        prev_obs = hist_observations[data_idx, 0]
        curr_obs = hist_observations[data_idx, 1]
        obs_diff = curr_obs - prev_obs
        if int(goal_indicate) == BEHIND:
            succ = (obs_diff[y_axis_idx_arr].max() >= SINGLE_STEP_MOVE_DIST).item()
        elif int(goal_indicate) == LEFT:
            succ = (obs_diff[x_axis_idx_arr].min() <= -SINGLE_STEP_MOVE_DIST).item()
        elif int(goal_indicate) == FRONT:
            succ = (obs_diff[y_axis_idx_arr].min() <= -SINGLE_STEP_MOVE_DIST).item()
        elif int(goal_indicate) == RIGHT:
            succ = (obs_diff[x_axis_idx_arr].max() >= SINGLE_STEP_MOVE_DIST).item()
        elif int(goal_indicate) == STILL:
            succ = (obs_diff.abs().max() < still_dist_threshold).item()
        else:
            raise NotImplementedError
        done_list.append(done)
        succ_list.append(succ)

    done = np.array(done_list)
    succ = np.array(succ_list)

    judge_result = dict(
        done=done,
        succ=succ,
    )

    return judge_result


def task_level_judge(num_obj, observations, goals: list) -> dict:
    batch_size = len(goals)
    done_list = []
    succ_list = []
    dist = 0.13 * 2.5
    order_target_diff_arr = np.array([dist, 0.0])
    norm_order_target_diff_arr = order_target_diff_arr / np.linalg.norm(order_target_diff_arr)
    order_max_error_angle = np.pi / 6
    sort_cycle_r = dist
    for data_idx in range(batch_size):
        goal_indicate = goals[data_idx][num_obj]
        if int(goal_indicate) in [TAU_2, TAU_3]:
            if int(goal_indicate) == TAU_2:
                goal_arr_arr = goals[data_idx][num_obj + 1: (num_obj + 1) * (2 + 1)].reshape(-1, num_obj + 1)
            elif int(goal_indicate) == TAU_3:
                goal_arr_arr = goals[data_idx][num_obj + 1: (num_obj + 1) * (3 + 1)].reshape(-1, num_obj + 1)
            else:
                raise NotImplementedError
            succ = True
            for goal_arr in goal_arr_arr:
                succ = succ and _success(num_obj, observations[data_idx].reshape(1, -1), goal_arr.reshape(1, -1)).item()
                if not succ:
                    break
        elif int(goal_indicate) == ORDER:
            diff_arr = np.diff(observations[data_idx].reshape((num_obj, -1)), axis=0)
            norm_diff_arr = diff_arr / np.linalg.norm(diff_arr, axis=1).reshape(-1, 1)
            diff_dot_arr = np.dot(norm_diff_arr, norm_order_target_diff_arr)
            succ = (diff_dot_arr >= np.cos(order_max_error_angle)).all().item()
        elif int(goal_indicate) == SORT:
            obs_xy_arr = observations[data_idx].reshape((num_obj, -1))
            cycle_center = obs_xy_arr[2]
            dist_list = []
            other_idx_list = [0, 1, 3, 4]
            for other_idx in other_idx_list:
                dist = np.linalg.norm(cycle_center - obs_xy_arr[other_idx])
                dist_list.append(dist)
            succ = (np.array(dist_list) < sort_cycle_r).all().item()
        else:
            raise NotImplementedError
        done_list.append(succ)
        succ_list.append(succ)

    done = np.array(done_list)
    succ = np.array(succ_list)

    judge_result = dict(
        done=done,
        succ=succ,
    )

    return judge_result


def terminal_fn(insts: List[str], observations: np.ndarray, **kwargs: Optional[Dict[str, Any]]):
    """Batched terminal function for CLEVR environment."""
    assert observations.ndim == 2, "observations must be [batch_size, obs_dim] shaped."
    assert "number_of_objects" in kwargs, "number_of_objects must be provided."
    assert "goals" in kwargs, "goals must be provided."

    num_obj = kwargs["number_of_objects"]
    goals = kwargs["goals"]

    assert isinstance(num_obj, int), "number_of_objects must be int."
    assert isinstance(goals, np.ndarray), "goals must be np.ndarray."
    assert goals.ndim == 2, "goals must be [batch_size, goal_dim] shaped."

    succ = _success(num_obj, observations, goals)

    return {
        "done": succ,
        "success": succ,
        "failure": np.zeros_like(succ, dtype=bool),
    }


def terminal_fn_with_level(insts: List[str], observations: np.ndarray, **kwargs: Optional[Dict[str, Any]]):
    """Batched terminal function for CLEVR environment."""
    assert observations.ndim == 2, "observations must be [batch_size, obs_dim] shaped."
    assert "number_of_objects" in kwargs, "number_of_objects must be provided."
    assert "goals" in kwargs, "goals must be provided."

    num_obj = kwargs["number_of_objects"]
    goals = kwargs["goals"]
    level = kwargs['level']

    assert isinstance(num_obj, int), "number_of_objects must be int."

    # assert isinstance(goals, np.ndarray), "goals must be np.ndarray."
    # assert goals.ndim == 2, "goals must be [batch_size, goal_dim] shaped."

    if level == 'tau_level':
        succ = _success(num_obj, observations, np.array(goals))
        done = succ
    elif level == 'step_level_a':
        hist_observations = kwargs['hist_observations']
        actions = kwargs['actions']
        judge_result = step_level_a_judge(num_obj=num_obj, observations=observations, hist_observations=hist_observations, actions=actions, goals=np.array(goals))
        done = judge_result['done']
        succ = judge_result['succ']
    elif level == 'step_level_s':
        hist_observations = kwargs['hist_observations']
        actions = kwargs['actions']
        judge_result = step_level_s_judge(num_obj=num_obj, observations=observations, hist_observations=hist_observations, actions=actions, goals=np.array(goals))
        done = judge_result['done']
        succ = judge_result['succ']
    elif level == 'task_level':
        judge_result = task_level_judge(num_obj=num_obj, observations=observations, goals=goals)
        done = judge_result['done']
        succ = judge_result['succ']
    else:
        raise NotImplementedError

    return {
        "done": done,
        "success": succ,
        "failure": np.zeros_like(succ, dtype=bool),
    }


def single_obs_to_state(observation: np.ndarray):
    state = np.zeros(69)
    # 规律：两个obs的维度，紧跟一个-0.195和一个1，然后紧跟3个0
    for i in range(0, 5):
        state[i*7 : i*7+2] = observation[ 2*i : 2*i+2 ]
        state[i*7 +2] = -0.195
        state[i*7 +3] = 1
        state[i*7+4 : i*7+7] = 0
    return state


if __name__ == "__main__":
    from operator import itemgetter
    from tqdm import tqdm

    np.random.seed(4)

    data_path = "../data/demo5_23-11-25_diverse.npy"
    limit = 1000000

    data = np.load(data_path, allow_pickle=True).item()
    observations, next_observations, goals, terminals = itemgetter(
        "observations", "next_observations", "goals", "terminals"
    )(data)

    indices = np.arange(len(observations))
    np.random.shuffle(indices)
    obs_accs = []
    next_obs_accs = []
    succ_cnt = 0
    term_early_cnt = 0
    invalid_data = {
        "observations": [],
        "next_observations": [],
        "goals": [],
        "terminals": [],
        "obs_pred_terms": [],
        "next_obs_pred_terms": [],
    }
    for idx in tqdm(indices[:limit], desc="eval"):
        obss = observations[idx]
        next_obss = next_observations[idx]
        terms = terminals[idx]

        init_obs = obss[0]
        goal = goals[idx][0]
        obs_acc = []
        next_obs_acc = []
        obs_terms = []
        next_obs_terms = []
        invalid = False
        for obs, next_obs, term in zip(obss, next_obss, terms):
            obs_t = terminal_fn(
                [None], np.array([obs]),
                {
                    "number_of_objects": 5,
                    "goals": np.array([goal]),
                    "initial_observations": np.array([init_obs]),
                }
            )
            next_obs_t = terminal_fn(
                [None], np.array([next_obs]),
                {
                    "number_of_objects": 5,
                    "goals": np.array([goal]),
                    "initial_observations": np.array([init_obs]),
                }
            )
            obs_terms.append(obs_t["done"][0].item())
            next_obs_terms.append(next_obs_t["done"][0].item())

            obs_acc.append(obs_t["done"][0].item() == False)
            next_obs_acc.append(next_obs_t["done"][0].item() == term.item())

            succ_cnt += obs_t["success"][0].item() + next_obs_t["success"][0].item()
            term_early_cnt += obs_t["done"][0].item()
            if obs_t["done"][0].item():
                invalid = True

            if term:
                break

        if invalid:
            invalid_data["observations"].append(obss[:len(obs_terms)].tolist())
            invalid_data["next_observations"].append(next_obss[:len(next_obs_terms)].tolist())
            invalid_data["goals"].append(goal.tolist())
            invalid_data["terminals"].append(terms[:len(obs_terms)].tolist())
            invalid_data["obs_pred_terms"].append(obs_terms)
            invalid_data["next_obs_pred_terms"].append(next_obs_terms)

        obs_accs.append(np.mean(obs_acc))
        next_obs_accs.append(np.mean(next_obs_acc))

    print("obs_acc: {:.4f}".format(np.mean(obs_accs)))
    print("next_obs_acc: {:.4f}".format(np.mean(next_obs_accs)))
    print("succ_cnt: {}".format(succ_cnt))
    print("term_early_cnt: {}".format(term_early_cnt))

    import pickle as pkl

    with open("invalid_data.pkl", "wb") as f:
        pkl.dump(invalid_data, f)

