# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The CLEVR-ROBOT environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random

from gym import spaces, utils, Env
import numpy as np

from clevr_robot_env.third_party.clevr_robot_env_utils.generate_question import generate_question_from_scene_struct
import clevr_robot_env.third_party.clevr_robot_env_utils.generate_scene as gs
import clevr_robot_env.third_party.clevr_robot_env_utils.question_engine as qeng

from clevr_robot_env.utils import load_utils
from clevr_robot_env.utils.xml_utils import convert_scene_to_xml

try:
    import clevr_robot_env.mujoco_env as mujoco_env  # custom mujoco_env
    from dm_control import mujoco
except ImportError as e:
    print(e)

file_dir = os.path.abspath(os.path.dirname(__file__))

DEFAULT_XML_PATH = os.path.join(file_dir, 'assets', 'clevr_default.xml')
FIXED_PATH = os.path.join(file_dir, 'templates', '10_fixed_objective.pkl')

# metadata
DEFAULT_METADATA_PATH = os.path.join(file_dir, 'metadata', 'metadata.json')
VARIABLE_OBJ_METADATA_PATH = os.path.join(file_dir, 'metadata',
                                          'variable_obj_meta_data.json')

# template_path
EVEN_Q_DIST_TEMPLATE = os.path.join(
    file_dir, 'templates/even_question_distribution.json')
VARIABLE_OBJ_TEMPLATE = os.path.join(file_dir, 'templates',
                                     'variable_object.json')
EVEN_Q_DIST_TEMPLATE_1 = os.path.join(
    file_dir, 'templates/even_question_distribution_1.json')

# fixed discrete action set
DIRECTIONS = [[1, 0], [0, 1], [-1, 0], [0, -1], [0.8, 0.8], [-0.8, 0.8],
              [0.8, -0.8], [-0.8, -0.8]]
X_RANGE, Y_RANGE = 0.7, 0.35


def _create_discrete_action_set():
    discrete_action_set = []
    for d in DIRECTIONS:
        for x in [-X_RANGE + i * X_RANGE / 5. for i in range(10)]:
            for y in [-Y_RANGE + i * 0.12 for i in range(10)]:
                discrete_action_set.append([[x, y], d])
    return discrete_action_set


DISCRETE_ACTION_SET = _create_discrete_action_set()

# cardinal vectors
# TODO: ideally this should be packaged into scene struct
four_cardinal_vectors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
four_cardinal_vectors = np.array(four_cardinal_vectors, dtype=np.float32)
four_cardinal_vectors_names = ['front', 'behind', 'left', 'right']


class ClevrEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """ClevrEnv."""

    def __init__(self,
                 maximum_episode_steps=100,
                 xml_path=None,
                 metadata_path=None,
                 template_path=None,
                 num_object=5,
                 agent_type='pm',
                 random_start=False,
                 description_num=15,
                 action_type='continuous',
                 obs_type='direct',
                 use_movement_bonus=False,
                 direct_obs=False,
                 reward_scale=1.0,
                 frame_skip=20,
                 shape_val=0.25,
                 min_move_dist=0.05,
                 resolution=64,
                 use_synonyms=False,
                 min_change_th=0.26,
                 use_polar=False,
                 use_subset_instruction=False,
                 systematic_generalization=False,
                 suppress_other_movement=False,
                 top_down_view=False,
                 variable_scene_content=False,
                 use_camera=False,
                 ):

        utils.EzPickle.__init__(self)
        initial_xml_path = DEFAULT_XML_PATH
        self.obj_name = []
        self.action_type = action_type
        self.use_movement_bonus = use_movement_bonus
        self.direct_obs = direct_obs
        self.obs_type = obs_type
        self.num_object = num_object
        self.variable_scene_content = variable_scene_content
        self.cache_valid_questions = variable_scene_content
        self.checker_board = variable_scene_content
        self.reward_scale = reward_scale
        self.shape_val = shape_val
        self.min_move_dist = min_move_dist
        self.res = resolution
        self.use_synonyms = use_synonyms
        self.min_change_th = min_change_th
        self.use_polar = use_polar
        self.suppress_other_movement = suppress_other_movement

        # loading meta data
        if metadata_path is None:
            metadata_path = DEFAULT_METADATA_PATH

        if self.variable_scene_content:
            print('loading variable input metadata')
            metadata_path = VARIABLE_OBJ_METADATA_PATH

        with open(metadata_path, 'r') as metadata_file:
            self.clevr_metadata = json.load(metadata_file)

        functions_by_name = {}
        for func in self.clevr_metadata['functions']:
            functions_by_name[func['name']] = func
        self.clevr_metadata['_functions_by_name'] = functions_by_name

        # information regarding question template
        if template_path is None:
            template_path = EVEN_Q_DIST_TEMPLATE
        if self.variable_scene_content:
            print('loading variable input template')
            template_path = VARIABLE_OBJ_TEMPLATE

        self.template_num = 0
        self.templates = {}
        fn = 'general_template'
        with open(template_path, 'r') as template_file:
            for i, template in enumerate(json.load(template_file)):
                self.template_num += 1
                key = (fn, i)
                self.templates[key] = template
        print('Read {} templates from disk'.format(self.template_num))

        # setting up camera transformation
        self.w2c, self.c2w = gs.camera_transformation_from_pose(90, -45)

        # sample a random scene and struct
        self.scene_graph, self.scene_struct = self.sample_random_scene()

        # total number of colors and shapes
        def one_hot_encoding(key_to_idx, max_length):
            encoding_map = {}
            for k in key_to_idx:
                one_hot_vector = [0] * max_length
                one_hot_vector[key_to_idx[k]] = 1
                encoding_map[k] = one_hot_vector
            return encoding_map

        mdata_types = self.clevr_metadata['types']
        self.color_n = len(mdata_types['Color'])
        self.color_to_idx = {c: i for i, c in enumerate(mdata_types['Color'])}
        self.color_to_one_hot = one_hot_encoding(self.color_to_idx, self.color_n)
        self.shape_n = len(mdata_types['Shape'])
        self.shape_to_idx = {s: i for i, s in enumerate(mdata_types['Shape'])}
        self.shape_to_one_hot = one_hot_encoding(self.shape_to_idx, self.shape_n)
        self.size_n = len(mdata_types['Size'])
        self.size_to_idx = {s: i for i, s in enumerate(mdata_types['Size'])}
        self.size_to_one_hot = one_hot_encoding(self.size_to_idx, self.size_n)
        self.mat_n = len(mdata_types['Material'])
        self.mat_to_idx = {s: i for i, s in enumerate(mdata_types['Material'])}
        self.mat_to_one_hot = one_hot_encoding(self.mat_to_idx, self.mat_n)

        # generate initial set of description from the scene graph
        self.description_num = description_num
        self.descriptions, self.full_descriptions = None, None
        self._update_description()
        self.obj_description = []
        self._update_object_description()

        mujoco_env.MujocoEnv.__init__(
            self,
            initial_xml_path,
            frame_skip,
            max_episode_steps=maximum_episode_steps,
            reward_threshold=0.,
            use_camera=use_camera,
        )

        # name of geometries in the scene
        self.obj_name = ['obj{}'.format(i) for i in range(self.num_object)]

        self.discrete_action_set = DISCRETE_ACTION_SET
        self.perfect_action_set = []
        for i in range(self.num_object):
            for d in DIRECTIONS:
                self.perfect_action_set.append(np.array([i] + d))

        # set discrete action space
        if self.action_type == 'discrete':
            self._action_set = DISCRETE_ACTION_SET
            self.action_space = spaces.Discrete(len(self._action_set))
        elif self.action_type == 'perfect':
            self._action_set = self.perfect_action_set
            self.action_space = spaces.Discrete(len(self._action_set))
        elif self.action_type == 'continuous':
            self.action_space = spaces.Box(
                low=-1.0, high=1.1, shape=[4], dtype=np.float32)
        elif self.action_type == 'collect':
            pass
        else:
            raise ValueError('{} is not a valid action type'.format(action_type))

        # setup camera and observation space
        if self.use_camera:
            self.camera = mujoco.MovableCamera(self.physics, height=480, width=480)
            
            camera_pose = self.camera.get_pose()
            lookat = [0.0, 0.0, 0.0]
            self.camera.set_pose(lookat, camera_pose.distance, 
                                 camera_pose.azimuth, camera_pose.elevation)
            self._top_down_view = top_down_view
            if top_down_view:
                camera_pose = self.camera.get_pose()
                self.camera.set_pose(camera_pose.lookat, camera_pose.distance,
                                     camera_pose.azimuth, -90)
            self.camera_setup()

        if self.direct_obs:
            self.observation_space = spaces.Box(
                low=np.concatenate(list(zip([-0.6] * num_object, [-0.4] * num_object))),
                high=np.concatenate(list(zip([0.6] * num_object, [0.6] * num_object))),
                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.res, self.res, 3), dtype=np.uint8)

        # agent type and randomness of starting location
        self.agent_type = agent_type
        self.random_start = random_start

        if not self.random_start:
            curr_scene_xml = convert_scene_to_xml(
                self.scene_graph,
                agent=self.agent_type,
                checker_board=self.checker_board)
        else:
            random_loc = '{} {} -0.2'.format(
                random.uniform(-0.6, 0.6), random.uniform(-0.3, 0.5))
            curr_scene_xml = convert_scene_to_xml(
                self.scene_graph,
                agent=self.agent_type,
                agent_start_loc=random_loc,
                checker_board=self.checker_board)
        self.load_xml_string(curr_scene_xml)

        self.valid_questions = []

        # counter for reset
        self.reset()
        self.curr_step = 0
        self.achieved_last_step = []
        self.achieved_last_step_program = []
        print('CLEVR-ROBOT environment initialized.')

    def step(self,
             a,
             record_achieved_goal=False,
             goal=None,
             atomic_goal=False,
             update_des=False
             ):
        # modified
        return self.get_obs(), 0, False, {}

    def teleport(self, loc):
        """Teleport the agent to loc."""
        # Location might be 2D because of no vertical movement
        curr_loc = self.get_body_com('point_mass')[:len(loc)]
        dsp_vec = loc - curr_loc
        qpos, qvel = self.physics.data.qpos.copy(), self.physics.data.qvel.copy()
        qpos[-2:] = qpos[-2:] + dsp_vec
        qvel[-2:] = np.zeros(2)
        self.set_state(qpos, qvel)

    def action_preprocessing(self, action) -> tuple:
        obj_selection = action // len(DIRECTIONS)
        dir_selection = action % len(DIRECTIONS)
        return obj_selection, dir_selection

    def step_discrete(self, a):
        """Take discrete step by teleporting and then push."""
        a = int(a)
        action = self.discrete_action_set[a]
        new_loc = np.array(action[0])
        self.teleport(new_loc)
        self.do_simulation(np.array(action[1]) * 1.1, int(self.frame_skip * 2.0))

    def step_perfect_noi(self, a):
        """Take a perfect step by teleporting and then push in fixed obj setting."""
        a = int(a)
        action = self._action_set[a]
        obj = action[0]
        obj_loc = self.get_body_com(self.obj_name[int(obj)])
        push_start = np.array(obj_loc)[:-1] - 0.15 * action[1:]
        dsp_vec = push_start - self.get_body_com('point_mass')[:-1]
        qpos, qvel = self.physics.data.qpos.copy(), self.physics.data.qvel.copy()
        qpos[-2:] = qpos[-2:] + dsp_vec
        qvel[-2:] = np.zeros(2)
        self.set_state(qpos, qvel)
        self.do_simulation(action[1:] * 1.0, int(self.frame_skip * 2.0))

    def step_perfect_oi(self, a):
        """Take a perfect step by teleporting and then push in fixed obj setting."""
        a = self.action_preprocessing(a)
        obj_selection, dir_selection = int(a[0]), int(a[1])
        direction = np.array(DIRECTIONS[dir_selection])
        obj_loc = self.scene_graph[obj_selection]['3d_coords'][:-1]
        push_start = np.array(obj_loc) - 0.15 * direction
        dsp_vec = push_start - self.get_body_com('point_mass')[:-1]
        qpos, qvel = self.physics.data.qpos.copy(), self.physics.data.qvel.copy()
        qpos[-2:] = qpos[-2:] + dsp_vec
        qvel[-2:] = np.zeros(2)
        self.set_state(qpos, qvel)
        self.do_simulation(direction * 1.0, int(self.frame_skip * 2.0))

    def step_continuous(self, a):
        """Take a continuous version of step discrete."""
        a = np.squeeze(a)
        x, y, theta, r = a[0] * 0.7, a[1] * 0.7, a[2] * np.pi, a[3]
        direction = np.array([np.cos(theta), np.sin(theta)]) * 1.2
        duration = int((r + 1.0) * self.frame_skip * 3.0)
        new_loc = np.array([x, y])
        qpos, qvel = self.physics.data.qpos, self.physics.data.qvel
        qpos[-2:], qvel[-2:] = new_loc, np.zeros(2)
        self.set_state(qpos, qvel)
        curr_loc = self.get_body_com('point_mass')
        dist = [curr_loc - self.get_body_com(name) for name in self.obj_name]
        dist = np.min(np.linalg.norm(dist, axis=1))
        self.do_simulation(direction, duration)

    def reset(self, new_scene_content=True):
        """Reset with a random configuration."""
        if new_scene_content or not self.variable_scene_content:
            # sample a random scene and struct
            self.scene_graph, self.scene_struct = self.sample_random_scene()
        else:
            # randomly perturb existing objects in the scene
            new_graph = gs.randomly_perturb_objects(self.scene_struct,
                                                    self.scene_graph)
            self.scene_graph = new_graph
            self.scene_struct['objects'] = self.scene_graph
            self.scene_struct['relationships'] = gs.compute_relationship(
                self.scene_struct)

        # Generate initial set of description from the scene graph.
        self.descriptions, self.full_descriptions = None, None
        self._update_description()
        self.curr_step = 0

        if not self.random_start:
            curr_scene_xml = convert_scene_to_xml(
                self.scene_graph,
                agent=self.agent_type,
                checker_board=self.checker_board)
        else:
            random_loc = '{} {} -0.2'.format(
                random.uniform(-0.6, 0.6), random.uniform(-0.3, 0.5))
            curr_scene_xml = convert_scene_to_xml(
                self.scene_graph,
                agent=self.agent_type,
                agent_start_loc=random_loc,
                checker_board=self.checker_board)
        self.load_xml_string(curr_scene_xml)

        if self.variable_scene_content and self.cache_valid_questions and new_scene_content:
            self.valid_questions = self.sample_valid_questions(100)
            if len(self.valid_questions) < 5:
                print('rerunning reset because valid question count is small')
                return self.reset(True)

        self._update_object_description()

        return self.get_obs()

    def get_obs(self):
        """Returns the state representation of the current scene."""
        if self.direct_obs and self.obs_type != 'order_invariant':
            return self.get_direct_obs()
        elif self.direct_obs and self.obs_type == 'order_invariant':
            return self.get_order_invariant_obs()
        else:
            return self.get_image_obs()

    def get_direct_obs(self):
        """Returns the direct state observation."""
        all_pos = np.array([self.get_body_com(name) for name in self.obj_name])
        has_obj = len(all_pos.shape) > 1
        all_pos = all_pos[:, :-1] if has_obj else np.zeros(2 * self.num_object)
        return all_pos.flatten()

    def get_image_obs(self):
        """Returns the image observation."""
        frame = self.render(mode='rgb_array')
        # frame = cv2.resize( frame, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite("/home/pangjc/zhangjj/render.png",  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # print(frame.shape)
        # return frame/255.
        return frame

    def get_order_invariant_obs(self):
        """Returns the order invariant observation.

    The returned vector will be a 2D array where the first axis is the object
    in the scene (which can be varying) and the second axis is the object
    description. Each object's description contains its x-y location and
    one-hot representation of its attributes (color, shape etc).
    """
        obs = []
        for obj in self.scene_graph:
            obj_vec = list(obj['3d_coords'][:-1])
            obj_vec += self.size_to_one_hot[obj['size']]
            obj_vec += self.color_to_one_hot[obj['color']]
            obj_vec += self.mat_to_one_hot[obj['material']]
            obj_vec += self.shape_to_one_hot[obj['shape']]
            obs.append(obj_vec)
        return np.array(obs)

    def get_achieved_goals(self):
        """Get goal that are achieved from the latest interaction."""
        return self.achieved_last_step

    def get_achieved_goal_programs(self):
        """Get goal programs that are achieved from the latest interaction."""
        return self.achieved_last_step_program

    def set_goal(self, goal_text, goal_program):
        """Set the goal to be used in standard RL settings."""
        raise NotImplementedError

    def sample_random_scene(self):
        """Sample a random scene base on current viewing angle."""
        if self.variable_scene_content:
            return gs.generate_scene_struct(self.c2w, self.num_object,
                                            self.clevr_metadata)
        else:
            return gs.generate_scene_struct(self.c2w, self.num_object)

    def sample_goal(self):
        """Sample a currently false statement and its corresponding text."""
        raise NotImplementedError

    def sample_random_action(self):
        """Sample a random action for the environment."""
        if self.obs_type == 'order_invariant' and self.action_type == 'perfect':
            action = [
                np.random.randint(low=0, high=self.num_object),
                np.random.randint(low=0, high=len(DIRECTIONS))
            ]
            return np.array(action)
        else:
            return self.action_space.sample()

    def sample_valid_questions(self, iterations=50):
        """Sample valid questions for the current scene content."""
        current_graph = self.scene_graph
        all_q = []
        for _ in range(iterations):
            new_graph = gs.randomly_perturb_objects(self.scene_struct, current_graph)
            self.scene_struct['objects'] = new_graph
            self.scene_struct['relationships'] = gs.compute_relationship(
                self.scene_struct)
            self._update_description()
            all_q += self.full_descriptions
        for q in all_q:
            for node in q['program']:
                if '_output' in node:
                    del node['_output']
        # get question that are unique and can be satisfied
        unique_and_feasible = {}
        for q in all_q:
            q_is_unique = repr(q['program']) not in unique_and_feasible
            if q['answer'] is True and q_is_unique:
                unique_and_feasible[repr(q['program'])] = q
        valid_q = []
        for q in unique_and_feasible:
            valid_q.append((unique_and_feasible[q]['question'],
                            unique_and_feasible[q]['program']))
        self.scene_struct['objects'] = current_graph
        self.scene_struct['relationships'] = gs.compute_relationship(
            self.scene_struct)
        return valid_q

    def answer_question(self, program, all_outputs=False):
        """Answer a functional program on the current scene."""
        return qeng.answer_question({'nodes': program},
                                    self.clevr_metadata,
                                    self.scene_struct,
                                    cache_outputs=False,
                                    all_outputs=all_outputs)

    def convert_order_invariant_to_direct(self, order_invariant_obs):
        """Converts the order invariant observation to state observation."""
        return order_invariant_obs[:, :2].flatten()

    def load_xml_string(self, xml_string):
        """Load the model into physics specified by a xml string."""
        self.physics.reload_from_xml_string(xml_string)

    def load_xml_path(self, xml_path):
        """Load the model into physics specified by a xml path."""
        self.physics.reload_from_xml_path(xml_path)

    def get_description(self):
        """Update and return the current scene description."""
        self._update_description()
        return self.descriptions, self.full_descriptions

    def _update_description(self, custom_n=None):
        """Update the text description of the current scene."""
        gq = generate_question_from_scene_struct
        dn = self.description_num if not custom_n else custom_n
        tn = self.template_num
        """
    self.descriptions, self.full_descriptions = gq(
        self.scene_struct,
        self.clevr_metadata,
        self.templates,
        templates_per_image=tn,
        instances_per_template=dn,
        use_synonyms=self.use_synonyms)
    """

    def _update_scene(self):
        """Update the scene description of the current scene."""
        self.previous_scene_graph = self.scene_graph
        for i, name in enumerate(self.obj_name):
            self.scene_graph[i]['3d_coords'] = tuple(self.get_body_com(name))
        self.scene_struct['objects'] = self.scene_graph
        self.scene_struct['relationships'] = gs.compute_relationship(
            self.scene_struct, use_polar=self.use_polar)

    def _update_object_description(self):
        """Update the scene description of the current scene."""
        self.obj_description = []
        for i in range(len(self.obj_name)):
            obj = self.scene_graph[i]
            color = obj['color']
            shape = obj['shape_name']
            material = obj['material']
            self.obj_description.append(' '.join([color, material, shape]))

    def _get_atomic_object_movements(self, displacement):
        """Get a list of sentences that describe the movements of object."""
        atomic_sentence = []
        for o, d in zip(self.obj_description, displacement):
            # TODO: this might need to be removed for stacking
            d_norm = np.linalg.norm(d[:-1])  # not counting height in displacement
            if d_norm > self.min_move_dist:
                max_d = np.argmax(np.dot(four_cardinal_vectors, d))
                atomic_sentence.append(' '.join(
                    [o, 'to', four_cardinal_vectors_names[max_d]]))
        return atomic_sentence

    def _get_fixed_object(self, answer):
        """Get the index and location of object that should be fixed in a query."""
        index, loc = -1, None
        for i, a in enumerate(answer):
            if a is True:
                index = random.choice(answer[i - 1])
            elif isinstance(a, float) or isinstance(a, int):
                index = answer[i]
                break
        if index >= 0:
            loc = np.array(self.scene_graph[index]['3d_coords'])[:-1]
        return index, loc

    def _get_obj_movement_bonus(self, fixed_obj_idx, displacement_vector):
        """Get the bonus reward for not moving other object."""
        del fixed_obj_idx
        norm = np.linalg.norm(displacement_vector, axis=-1)
        total_norm = norm.sum()
        return 0.5 * np.exp(-total_norm * 7)

    def _reward(self):
        # modified
        raise 0


import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from clevr_utils import goal_src, goal_dst
from clevr_utils import color_list


class LangEnv(ClevrEnv):
    def __init__(self, maximum_episode_steps=100, xml_path=None, metadata_path=None, template_path=None, num_object=5,
                 agent_type='pm', random_start=False, description_num=15, action_type='continuous', obs_type='order_invariant',
                 use_movement_bonus=False, direct_obs=True, reward_scale=1, frame_skip=20, shape_val=0.25,
                 min_move_dist=0.05, resolution=64, use_synonyms=False, min_change_th=0.26, use_polar=False,
                 use_subset_instruction=False, systematic_generalization=False, suppress_other_movement=False,
                 top_down_view=False, variable_scene_content=False,
                 use_camera=False):
        self.max_goal_length = 32

        self.dist_scale = 10
        self.success_reward = 100
        self.fail_reward = -10
        
        # self.success_dist_threshold = 0.13 * 2 + 0.13 * 2
        
        self.success_dist_threshold = 0.13 * 2 + 0.13 * 1
        
        self.fail_dist_threshold = 0.20
        
        self.min_punish_dist = 0.1

        self.goal_count = 2
        self.goal_src = goal_src
        self.goal_dst = goal_dst
        self.init_goal(num_object)
        
        self.reward_info_template = {
            'result': 0,
            'near': 0,
            'still': 0,
            'second_near': 0,
        }
        self.reward_info_list = None
        self.bias_list = None
        self.dist_list = None
        self.prev_dist = None
        self.init_obs = None

        super().__init__(maximum_episode_steps, xml_path, metadata_path, template_path, num_object, agent_type,
                         random_start, description_num, action_type, obs_type, use_movement_bonus, direct_obs,
                         reward_scale, frame_skip, shape_val, min_move_dist, resolution, use_synonyms, min_change_th,
                         use_polar, use_subset_instruction, systematic_generalization, suppress_other_movement,
                         top_down_view, variable_scene_content,
                         use_camera=use_camera)
        assert self.direct_obs and self.obs_type == 'order_invariant'
        
        reset_result = self.reset()
        try:
            obs, _ = reset_result
        except:
            obs = reset_result
        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=obs.shape,
            dtype=np.float32,
        )
        self._action_set = self.perfect_action_set
        self.action_space = spaces.Discrete(len(self._action_set))

    def init_goal(self, num_object=None):
        if num_object is None:
            num_object = self.num_object
        assert isinstance(num_object, int)

        goal_idx_arr = np.random.choice(np.arange(num_object), self.goal_count, replace=False)
        self.goal_arr = np.zeros(num_object + int(issubclass(type(self), Lang1Env)), dtype=int)
        assert self.goal_count == 2
        src_idx, dst_idx = goal_idx_arr
        self.goal_arr[src_idx] = self.goal_src
        self.goal_arr[dst_idx] = self.goal_dst

    def get_obs(self) -> np.ndarray:
        obs = []
        start_idx = 0
        xy_len = 2
        qpos_len = 7
        for _ in range(self.num_object):
            obs.append(self.physics.data.qpos[start_idx: start_idx + xy_len])
            start_idx += qpos_len
        """
        for obj in self.scene_graph:
        # scene_graph -> xml -> MuJoCo -> state
        obj_vec = list(obj['3d_coords'][:-1])
        # obj_vec += self.size_to_one_hot[obj['size']]
        # obj_vec += self.color_to_one_hot[obj['color']]
        # obj_vec += self.mat_to_one_hot[obj['material']]
        # obj_vec += self.shape_to_one_hot[obj['shape']]
        obs.append(obj_vec)
        """

        obs.append(self.goal_arr.copy())

        return np.concatenate(obs)
    
    def step_collect(self, action):
        goal_src_idx = np.where(self.goal_arr[:self.num_object] == self.goal_src)[0].item()
        obj_selection = goal_src_idx
        dir_selection = action
        direction = np.array(DIRECTIONS[dir_selection])
        obj_loc = self.scene_graph[obj_selection]['3d_coords'][:-1]
        push_start = np.array(obj_loc) - 0.15 * direction
        dsp_vec = push_start - self.get_body_com('point_mass')[:-1]
        qpos, qvel = self.physics.data.qpos.copy(), self.physics.data.qvel.copy()
        qpos[-2:] = qpos[-2:] + dsp_vec
        qvel[-2:] = np.zeros(2)
        self.set_state(qpos, qvel)
        self.do_simulation(direction * 1.0, int(self.frame_skip * 2.0))

    def step(self,
             a,
             record_achieved_goal=False,
             goal=None,
             atomic_goal=False,
             update_des=False
             ):
        """Take step a in the environment."""

        info = {}

        if not self.obj_name:
            self.do_simulation([0, 0], self.frame_skip)
            return self.get_obs(), 0, False, None

        curr_state = np.array([self.get_body_com(name) for name in self.obj_name])

        if self.action_type == 'discrete':
            self.step_discrete(a)
        elif self.action_type == 'perfect' and self.obs_type != 'order_invariant':
            self.step_perfect_noi(a)
        elif self.action_type == 'perfect' and self.obs_type == 'order_invariant':
            self.step_perfect_oi(a)
        elif self.action_type == 'continuous':
            self.step_continuous(a)
        elif self.action_type == 'collect':
            self.step_collect(a)

        new_state = np.array([self.get_body_com(name) for name in self.obj_name])
        displacement_vector = np.stack(
            [a - b for a, b in zip(curr_state, new_state)])
        atomic_movement_description = self._get_atomic_object_movements(
            displacement_vector)

        self._update_scene()
        if update_des:
            self._update_description()
            info['descriptions'] = self.descriptions
            info['full_descriptions'] = self.full_descriptions

        if record_achieved_goal and atomic_goal:
            self.achieved_last_step += atomic_movement_description

        assert not goal
        self.curr_step += 1
        r = self._reward()

        is_success = self.is_success()
        is_fail = self.is_fail()
        episode_flag = self.curr_step >= self.max_episode_steps
        done = is_success or is_fail or episode_flag
        info['is_fail'] = is_fail
        info['is_success'] = not is_fail and is_success

        if done:
            custom_info = {}
            # custom_info['bias_min'] = np.min(self.bias_list)
            # custom_info['bias_mean'] = np.mean(self.bias_list)
            # custom_info['bias_max'] = np.max(self.bias_list)
            
            custom_info['dist_min'] = np.min(self.dist_list)
            custom_info['dist_mean'] = np.mean(self.dist_list)
            custom_info['dist_max'] = np.max(self.dist_list)
            
            for key in self.reward_info_template.keys():
                if key == 'result':
                    custom_info[f'{key}_mean'] = np.mean([reward_info[key] for reward_info in self.reward_info_list])
                else:
                    custom_info[f'{key}_min'] = np.min([reward_info[key] for reward_info in self.reward_info_list])
                    custom_info[f'{key}_mean'] = np.mean([reward_info[key] for reward_info in self.reward_info_list])
                    custom_info[f'{key}_max'] = np.max([reward_info[key] for reward_info in self.reward_info_list])
            
            info['custom_info'] = custom_info

        self.prev_dist = self.compute_dist()

        obs = self.get_obs()

        return obs, r, done, info

    def sample_random_action(self):
        return self.action_space.sample()

    def reset_mdp_utils(self):
        self.curr_step = 0
        self.init_goal()

        obs = self.get_obs()
        self.reward_info_list = []
        self.bias_list = []
        self.dist_list = []
        self.prev_dist = self.compute_dist()
        self.init_obs = obs.copy()

        self.reward_info_list.append(self.reward_info_template.copy())
        self.bias_list.append(0)
        self.dist_list.append(self.prev_dist)

        return obs

    def reset(self, new_scene_content=True):
        """Reset with a random configuration."""
        if new_scene_content or not self.variable_scene_content:
            # sample a random scene and struct
            self.scene_graph, self.scene_struct = self.sample_random_scene()
        else:
            # randomly perturb existing objects in the scene
            new_graph = gs.randomly_perturb_objects(self.scene_struct,
                                                    self.scene_graph)
            self.scene_graph = new_graph
            self.scene_struct['objects'] = self.scene_graph
            self.scene_struct['relationships'] = gs.compute_relationship(
                self.scene_struct)

        # Generate initial set of description from the scene graph.
        self.descriptions, self.full_descriptions = None, None
        self._update_description()

        if not self.random_start:
            curr_scene_xml = convert_scene_to_xml(
                self.scene_graph,
                agent=self.agent_type,
                checker_board=self.checker_board)
        else:
            random_loc = '{} {} -0.2'.format(
                random.uniform(-0.6, 0.6), random.uniform(-0.3, 0.5))
            curr_scene_xml = convert_scene_to_xml(
                self.scene_graph,
                agent=self.agent_type,
                agent_start_loc=random_loc,
                checker_board=self.checker_board)
        self.load_xml_string(curr_scene_xml)

        self._update_object_description()

        obs = self.reset_mdp_utils()

        return obs

    def compute_dist(self):
        obs = self.get_obs()

        obs_no_goal = obs[:2 * self.num_object]
        reshaped_obs = obs_no_goal.reshape((self.num_object, -1))

        goal_src_idx = np.where(self.goal_arr[:self.num_object] == self.goal_src)[0].item()
        goal_dst_idx = np.where(self.goal_arr[:self.num_object] == self.goal_dst)[0].item()

        goal_src_xy = reshaped_obs[goal_src_idx]
        goal_dst_xy = reshaped_obs[goal_dst_idx]

        dist = np.linalg.norm(goal_src_xy - goal_dst_xy)

        return dist

    def is_success(self):
        curr_dist = self.compute_dist()

        is_success = curr_dist < self.success_dist_threshold

        return is_success

    def compute_bias(self) -> np.ndarray:
        init_obs = self.init_obs.copy()
        curr_obs = self.get_obs()

        init_obs_no_goal = init_obs[:2 * self.num_object]
        curr_obs_no_goal = curr_obs[:2 * self.num_object]
        shaped_init_obs = init_obs_no_goal.reshape((self.num_object, -1))
        shaped_curr_obs = curr_obs_no_goal.reshape((self.num_object, -1))

        goal_src_idx = np.where(self.goal_arr[:self.num_object] == self.goal_src)[0].item()

        bias_arr = shaped_curr_obs - shaped_init_obs
        bias_arr[goal_src_idx] = 0

        bias = np.linalg.norm(bias_arr, axis=1)

        new_bias = np.where(bias >= self.min_punish_dist, bias, 0)

        return new_bias

    def is_fail(self):
        bias = self.compute_bias()

        is_fail = np.any(bias > self.fail_dist_threshold)

        return is_fail

    def _reward(self):
        prev_dist = self.prev_dist
        curr_dist = self.compute_dist()
        bias = self.compute_bias()
        bias_max = np.max(bias)
        reward_info = self.reward_info_template.copy()

        if self.is_fail():
            reward = self.fail_reward
            reward_info['result'] = self.fail_reward
        elif self.is_success():
            reward = self.success_reward
            reward_info['result'] = self.success_reward
        else:
            near_reward = (prev_dist - curr_dist) * self.dist_scale
            still_reward = -bias_max * self.dist_scale * 0.1

            reward = near_reward + still_reward
            
            reward_info['near'] = near_reward
            reward_info['still'] = still_reward

        self.reward_info_list.append(reward_info)
        self.bias_list.append(bias_max.item())
        self.dist_list.append(self.prev_dist)

        return reward


class DemoEnv(LangEnv):
    def compute_bias(self) -> np.ndarray:
        return np.zeros(self.num_object)


BEHIND = 0  # +y
LEFT = 1  # -x
FRONT = 2  # -y
RIGHT = 3  # +x
dir_list = [
    BEHIND,
    LEFT,
    FRONT,
    RIGHT,
]
dir_cnt = len(dir_list)
step_size = 360 / dir_cnt
rg_size = step_size / 2
dir2angle_rg = {
    0: np.array([90 - rg_size, 90 + rg_size]),
}
for dir in range(1, dir_cnt):
    dir2angle_rg[dir] = (dir2angle_rg[dir - 1] + step_size) % 360


def rotate(origin: np.ndarray, point: np.ndarray, angle: np.float64) -> np.ndarray:
    assert origin.size == 2
    assert point.size == 2
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.array([qx, qy])


class Lang1Env(LangEnv):
    def __init__(self, maximum_episode_steps=100, xml_path=None, metadata_path=None, template_path=None, num_object=5,
                 agent_type='pm', random_start=False, description_num=15, action_type='continuous', obs_type='direct',
                 use_movement_bonus=False, direct_obs=False, reward_scale=1, frame_skip=20, shape_val=0.25,
                 min_move_dist=0.05, resolution=64, use_synonyms=False, min_change_th=0.26, use_polar=False,
                 use_subset_instruction=False, systematic_generalization=False, suppress_other_movement=False,
                 top_down_view=False, variable_scene_content=False,
                 use_camera=False):
        # template_path = EVEN_Q_DIST_TEMPLATE_1

        self.epsilon = 1e-3
        self.dir = None
        self.delta_xy = None

        super().__init__(maximum_episode_steps, xml_path, metadata_path, template_path, num_object, agent_type,
                         random_start, description_num, action_type, obs_type, use_movement_bonus, direct_obs,
                         reward_scale, frame_skip, shape_val, min_move_dist, resolution, use_synonyms, min_change_th,
                         use_polar, use_subset_instruction, systematic_generalization, suppress_other_movement,
                         top_down_view, variable_scene_content,
                         use_camera=use_camera)

        self.init_goal(num_object)

    def update_delta_xy(self, dir: int, is_second: bool = False) -> np.ndarray:
        assert dir in dir2angle_rg.keys(), f'{dir} not in {dir2angle_rg.keys()}'

        if is_second:
            delta_xy = np.array([0, 0.4 + self.success_dist_threshold])
        else:
            delta_xy = np.array([0, self.success_dist_threshold / 2])
            
        rotate_angle = (360 / dir_cnt) / 180 * np.pi

        origin = np.zeros(2)
        for _ in range(dir):
            delta_xy = rotate(origin=origin, point=delta_xy, angle=rotate_angle)
        delta_xy = np.where(np.abs(delta_xy) >= self.epsilon, delta_xy, 0)
        
        return delta_xy.copy()

    def get_goal_xy(self, indicate: int):
        assert indicate in [self.goal_src, self.goal_dst]
        obs = self.get_obs()
        shaped_xy_arr = obs[:2 * self.num_object].reshape(self.num_object, -1)

        target_goal_idx = np.where(self.goal_arr[:self.num_object] == indicate)[0].item()

        return shaped_xy_arr[target_goal_idx].copy()

    def reset_mdp_utils(self, dir=None):
        if dir is None:
            self.dir = np.random.choice(dir_list).item()
        else:
            self.dir = dir
        self.delta_xy = self.update_delta_xy(dir=self.dir)

        obs = super().reset_mdp_utils()
        self.goal_arr[self.num_object] = self.dir
        obs = self.get_obs()
        self.init_obs = obs.copy()

        goal_dst_xy = self.get_goal_xy(self.goal_dst).copy()
        angle_rg = dir2angle_rg[self.dir].copy()
        if angle_rg[1] < angle_rg[0]:
            angle_rg_mean = 0
        else:
            angle_rg_mean = np.mean(angle_rg).item()
        assert abs(self.compute_angle(goal_dst_xy + self.delta_xy) - angle_rg_mean) < self.epsilon

        return obs

    def reset(self, new_scene_content=True, dir=None):
        """Reset with a random configuration."""
        if new_scene_content or not self.variable_scene_content:
            # sample a random scene and struct
            self.scene_graph, self.scene_struct = self.sample_random_scene()
        else:
            # randomly perturb existing objects in the scene
            new_graph = gs.randomly_perturb_objects(self.scene_struct,
                                                    self.scene_graph)
            self.scene_graph = new_graph
            self.scene_struct['objects'] = self.scene_graph
            self.scene_struct['relationships'] = gs.compute_relationship(
                self.scene_struct)

        # Generate initial set of description from the scene graph.
        self.descriptions, self.full_descriptions = None, None
        self._update_description()

        if not self.random_start:
            curr_scene_xml = convert_scene_to_xml(
                self.scene_graph,
                agent=self.agent_type,
                checker_board=self.checker_board)
        else:
            random_loc = '{} {} -0.2'.format(
                random.uniform(-0.6, 0.6), random.uniform(-0.3, 0.5))
            curr_scene_xml = convert_scene_to_xml(
                self.scene_graph,
                agent=self.agent_type,
                agent_start_loc=random_loc,
                checker_board=self.checker_board)
        self.load_xml_string(curr_scene_xml)

        self._update_object_description()

        obs = self.reset_mdp_utils(dir=dir)

        return obs

    def compute_dist(self, target_xy: np.ndarray = None):
        if target_xy is None:
            goal_dst_xy = self.get_goal_xy(self.goal_dst).copy()
            target_xy = goal_dst_xy + self.delta_xy
        assert isinstance(target_xy, np.ndarray)

        goal_src_xy = self.get_goal_xy(self.goal_src).copy()

        dist = np.linalg.norm(goal_src_xy - target_xy)

        return dist

    def compute_angle(self, src_xy: np.ndarray, dst_xy: np.ndarray = None):
        if dst_xy is None:
            dst_xy = self.get_goal_xy(self.goal_dst).copy()

        delta_xy = src_xy - dst_xy
        delta_xy = np.where(np.abs(delta_xy) >= self.epsilon, delta_xy, 0)

        cos_value = delta_xy[0] / np.linalg.norm(delta_xy)
        negative_flag = delta_xy[1] < 0
        angle = np.arccos(cos_value)
        if negative_flag:
            angle = 2 * np.pi - angle

        _angle = angle * 180 / np.pi

        return _angle

    def calculate_description(self, next_obs: np.ndarray, goal: np.ndarray, data_type="new") -> int:
        shape_obs = next_obs.reshape(self.num_object, -1)
        
        if data_type == "new":
            src_idx = int(np.where(goal[:self.num_object] == self.goal_src)[0])
            dst_idx = int(np.where(goal[:self.num_object] == self.goal_dst)[0])
        else:
            src_idx = int(np.where(goal[:self.num_object] == 1)[0][0])
            dst_idx = int(np.where(goal[:self.num_object] == 1)[0][-1])
        src_xy = shape_obs[src_idx]
        dst_xy = shape_obs[dst_idx]
        
        curr_angle = self.compute_angle(src_xy, dst_xy)
        orientation = None
        for dir in range(dir_cnt):
            angle_rg = dir2angle_rg[dir]
            if angle_rg[1] < angle_rg[0]:
                angle_flag = (angle_rg[0] <= curr_angle < 360) or (0 <= curr_angle < angle_rg[1])
            else:
                angle_flag = angle_rg[0] <= curr_angle < angle_rg[1]
            if angle_flag:
                orientation = dir
        assert orientation is not None
        
        return src_idx, dst_idx, orientation

    def compute_dir(self, src_color: str, dst_color: str) -> int:
        obs = self.get_obs()
        shaped_xy_arr = obs[:2 * self.num_object].reshape(self.num_object, -1)
        
        src_idx = color_list.index(src_color)
        dst_idx = color_list.index(dst_color)
        src_xy = shaped_xy_arr[src_idx]
        dst_xy = shaped_xy_arr[dst_idx]
               
        curr_angle = self.compute_angle(src_xy, dst_xy)
        dir = None
        for dir_candidate in range(dir_cnt):
            angle_rg = dir2angle_rg[dir_candidate]
            if angle_rg[1] < angle_rg[0]:
                angle_flag = (angle_rg[0] <= curr_angle < 360) or (0 <= curr_angle < angle_rg[1])
            else:
                angle_flag = angle_rg[0] <= curr_angle < angle_rg[1]
            if angle_flag:
                dir = dir_candidate
        assert dir is not None
        
        return dir

    def is_success(self):
        goal_src_xy = self.get_goal_xy(self.goal_src).copy()
        goal_dst_xy = self.get_goal_xy(self.goal_dst).copy()
        curr_angle = self.compute_angle(goal_src_xy)

        curr_dist = self.compute_dist(goal_dst_xy)

        angle_rg = dir2angle_rg[self.dir]
        if angle_rg[1] < angle_rg[0]:
            angle_flag = (angle_rg[0] <= curr_angle < 360) or (0 <= curr_angle < angle_rg[1])
        else:
            angle_flag = angle_rg[0] <= curr_angle < angle_rg[1]
        dist_flag = curr_dist < self.success_dist_threshold

        is_success = angle_flag and dist_flag

        return is_success


class Lang2Env(Lang1Env):
    def __init__(self, maximum_episode_steps=100, xml_path=None, metadata_path=None, template_path=None, num_object=5,
                 agent_type='pm', random_start=False, description_num=15, action_type='continuous', obs_type='direct',
                 use_movement_bonus=False, direct_obs=False, reward_scale=1, frame_skip=20, shape_val=0.25,
                 min_move_dist=0.05, resolution=64, use_synonyms=False, min_change_th=0.26, use_polar=False,
                 use_subset_instruction=False, systematic_generalization=False, suppress_other_movement=False,
                 top_down_view=False, variable_scene_content=False,
                 fail_dist=0.20,
                 use_camera=False):
        super().__init__(maximum_episode_steps, xml_path, metadata_path, template_path, num_object, agent_type,
                         random_start, description_num, action_type, obs_type, use_movement_bonus, direct_obs,
                         reward_scale, frame_skip, shape_val, min_move_dist, resolution, use_synonyms, min_change_th,
                         use_polar, use_subset_instruction, systematic_generalization, suppress_other_movement,
                         top_down_view, variable_scene_content,
                         use_camera=use_camera)
        self.fail_dist_threshold = fail_dist
        
        if self.action_type == 'collect':
            self.perfect_action_set = []
            for d in DIRECTIONS:
                self.perfect_action_set.append(np.array(d))
            self._action_set = self.perfect_action_set
            self.action_space = spaces.Discrete(len(self._action_set))

    def compute_bias(self) -> np.ndarray:
        return np.zeros(self.num_object)


class LlataEnv(LangEnv):
    def step(self,
             a,
             record_achieved_goal=False,
             goal=None,
             atomic_goal=False,
             update_des=False
             ):
        obs, reward, done, info = super().step(a, record_achieved_goal, goal, atomic_goal, update_des)

        # reward and done is useless
        reward = 0
        done = False

        return obs, reward, done, info


# from utils import get_best_cuda
# import copy
# from stable_baselines3 import LangGCPPPO
# from utils import models_dir


# class LangGCPEnv(Lang2Env):
#     def __init__(self, maximum_episode_steps=100, xml_path=None, metadata_path=None, template_path=None, num_object=5,
#                  agent_type='pm', random_start=False, description_num=15, action_type='continuous', obs_type='direct',
#                  use_movement_bonus=False, direct_obs=False, reward_scale=1, frame_skip=20, shape_val=0.25,
#                  min_move_dist=0.05, resolution=64, use_synonyms=False, min_change_th=0.26, use_polar=False,
#                  use_subset_instruction=False, systematic_generalization=False, suppress_other_movement=False,
#                  top_down_view=False, variable_scene_content=False,
#                  fail_dist=0.20,
#                  train_template_cnt=9,
#                  language_model_type='onehot',
#                  mode='train',
#                  use_camera=False,
#                  ):
#         self.train_template_cnt = train_template_cnt
#         self.language_model_type = language_model_type
#         self.mode = mode
        
#         self.template_idx = None
#         self.orientation_idx = None
#         self.color_idx = None
        
#         self.total_template_arr = np.array(total_template_list)
#         self.train_template_arr = np.array(train_template_list)[:self.train_template_cnt]
#         self.error_template_arr = np.array(error_template_list)
#         self.test_template_arr = np.array(test_template_list)
        
#         self.total_orientation_arr = np.array(total_orientation_list)
#         self.orientation_arr = np.array(orientation_list)
        
#         self.color_arr = np.array(color_list)
        
#         if self.mode == 'train':
#             self.reset_template_arr = self.train_template_arr.copy()
#             self.reset_orientation_arr = self.orientation_arr.copy()
#         elif self.mode == 'test':
#             self.reset_template_arr = self.test_template_arr.copy()
#             self.reset_orientation_arr = self.orientation_arr.copy()
#         elif self.mode == 'error':
#             self.reset_template_arr = self.test_template_arr.copy()
#             self.reset_orientation_arr = self.total_orientation_arr.copy()
#         else:
#             raise NotImplementedError
        
#         # NL in str
#         self.language_goal = None

#         super().__init__(maximum_episode_steps, xml_path, metadata_path, template_path, num_object, agent_type,
#                          random_start, description_num, action_type, obs_type, use_movement_bonus, direct_obs,
#                          reward_scale, frame_skip, shape_val, min_move_dist, resolution, use_synonyms, min_change_th,
#                          use_polar, use_subset_instruction, systematic_generalization, suppress_other_movement,
#                          top_down_view, variable_scene_content,
#                          use_camera=use_camera,
#                          fail_dist=fail_dist)
#         self.goal_abs = None

#     def language_goal2self_goal(self, recons_color_idx_arr: np.ndarray, recons_orientation_idx: int):
#         self.goal_arr = np.zeros(self.num_object + 1)
        
#         assert self.goal_count == 2
#         assert recons_color_idx_arr.size == self.goal_count
#         goal_src_idx, goal_dst_idx = recons_color_idx_arr
#         self.goal_arr[goal_src_idx] = self.goal_src
#         self.goal_arr[goal_dst_idx] = self.goal_dst
#         self.goal_arr[-1] = recons_orientation_idx
        
#         self.dir = recons_orientation_idx
#         self.delta_xy = self.update_delta_xy(dir=self.dir)

#     def reset_mdp_utils(self, dir=None):
#         new_template_idx = np.random.choice(np.arange(self.reset_template_arr.size))
#         new_template = self.reset_template_arr[new_template_idx]
        
#         new_orientation_idx = np.random.choice(np.arange(self.reset_orientation_arr.size))
#         new_orientation = self.reset_orientation_arr[new_orientation_idx]

#         new_color_idx_arr = np.random.choice(np.arange(self.color_arr.size), self.goal_count, replace=False)

#         new_color_arr = self.color_arr[new_color_idx_arr]
        
#         language_goal = new_template.format(new_color_arr[0], new_orientation, new_color_arr[1])
#         self.language_goal = language_goal
        
#         onehot_template_idx = np.where(self.total_template_arr == new_template)[0].item()
#         onehot_orientation_idx = np.where(self.total_orientation_arr == new_orientation)[0].item()
        
#         self.template_idx = onehot_template_idx
#         self.orientation_idx = onehot_orientation_idx
#         self.color_idx = color_pair2color_idx[(new_color_idx_arr[0], new_color_idx_arr[1])]
                
#         obs = super().reset_mdp_utils(dir)
        
#         recons_color_idx_arr = new_color_idx_arr
#         if self.mode in ['train', 'test']:
#             recons_orientation_idx = new_orientation_idx
#         elif self.mode in ['error']:
#             recons_orientation_idx = new_orientation_idx // 3
#         else:
#             raise NotImplementedError
#         self.language_goal2self_goal(recons_color_idx_arr=recons_color_idx_arr, recons_orientation_idx=recons_orientation_idx)

#         self.goal_abs = None

#         return obs

#     def get_obs(self) -> np.ndarray:
#         obs_no_goal_arr = self.get_obs_no_goal()
        
#         if self.language_goal is None:
#             language_goal_arr = np.zeros(3)
#         else:
#             language_goal_arr = np.array([self.template_idx, self.orientation_idx, self.color_idx])

#         obs = np.r_[obs_no_goal_arr, language_goal_arr]

#         return obs

#     def get_obs_no_goal(self) -> np.ndarray:
#         obs = []
#         start_idx = 0
#         xy_len = 2
#         qpos_len = 7
#         for _ in range(self.num_object):
#             obs.append(self.physics.data.qpos[start_idx: start_idx + xy_len])
#             start_idx += qpos_len

#         return np.concatenate(obs)

#     def high_step_setup(self, goal_abs: np.ndarray) -> None:
#         self.goal_abs = goal_abs.copy()
        
#         super().reset_mdp_utils()
    
#     def high_get_obs(self, obs: np.ndarray):
#         obs_no_goal = obs[:2 * self.num_object].copy()
#         goal_abs = self.goal_abs.copy()
        
#         obs = np.r_[obs_no_goal, goal_abs]
        
#         return obs
    
#     def high_step(self, agent: LangGCPPPO, n_step: int = 5) -> None:
#         obs = self.get_obs()
#         high_obs = self.high_get_obs(obs)
#         reward = 0
#         done = False
#         info = {}
#         for step in np.arange(n_step):
#             action = agent.predict(high_obs, deterministic=False)[0]
#             obs, reward, done, info = self.step(action)
#             high_obs = self.high_get_obs(obs)
        
#         return obs, reward, done, info


# class LangRGBEnv(Lang2Env):
#     def get_frame(self) -> np.ndarray:
#         frame = self.render(mode='rgb_array')
#         frame = cv2.resize(frame, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC)
        
#         shape = frame.shape
#         frame = frame.reshape(tuple(reversed(shape)))
        
#         return frame
        
#     def get_rgb_obs(self) -> np.ndarray:
#         obs = []
#         start_idx = 0
#         xy_len = 2
#         qpos_len = 7
#         for _ in range(self.num_object):
#             obs.append(self.physics.data.qpos[start_idx: start_idx + xy_len])
#             start_idx += qpos_len

#         image_arr = self.get_image_obs()
        
#         rgb_shape = image_arr.shape
#         image_arr = image_arr.reshape(tuple(reversed(rgb_shape)))
        
#         rgb_obs = np.r_[image_arr.flatten(), self.goal_arr.copy()]
        
#         return rgb_obs
    
#     def reset(self, new_scene_content=True, dir=None):
#         super().reset(new_scene_content, dir)
        
#         obs = self.get_rgb_obs()
        
#         return obs
    
#     def step(self, a, record_achieved_goal=False, goal=None, atomic_goal=False, update_des=False):
#         _, reward, done, info = super().step(a, record_achieved_goal, goal, atomic_goal, update_des)
        
#         obs = self.get_rgb_obs()
        
#         return obs, reward, done, info

#     # def compute_bias(self) -> np.ndarray:
#     #     init_obs = self.init_obs.copy()
#     #     curr_obs = self.get_obs()

#     #     init_obs_no_goal = init_obs[:2 * self.num_object]
#     #     curr_obs_no_goal = curr_obs[:2 * self.num_object]
#     #     shaped_init_obs = init_obs_no_goal.reshape((self.num_object, -1))
#     #     shaped_curr_obs = curr_obs_no_goal.reshape((self.num_object, -1))

#     #     goal_src_idx = np.where(self.goal_arr[:self.num_object] == self.goal_src)[0].item()

#     #     bias_arr = shaped_curr_obs - shaped_init_obs
#     #     bias_arr[goal_src_idx] = 0

#     #     bias = np.linalg.norm(bias_arr, axis=1)

#     #     new_bias = np.where(bias >= self.min_punish_dist, bias, 0)

#     #     return new_bias


# from pathlib import Path


# class HighLangGCPEnv(Env):
#     def __init__(self,
#                  init_high=False,
#                  n_step=5,
#                  num_object=5,
#                  maximum_episode_steps=50,
#                  env_type='arrangement',
#                  language_model_type='onehot',
#                  agent_name=None,
#                  use_camera=False,
#                  ):
#         super().__init__()
        
#         self.init_high = init_high
#         self.n_step = n_step
#         self.maximum_episode_steps = maximum_episode_steps
#         self.env_type = env_type
#         self.language_model_type = language_model_type
#         self.agent_name = agent_name
        
#         if self.language_model_type == 'no_lm':
#             self.maximum_episode_steps = 50
        
#         self.fail_reward = -10
#         self.success_reward = 100
        
#         # ['red', 'blue', 'green', 'purple', 'cyan']
#         if self.env_type == 'arrangement':
#             target_satisfied_cnt = 9
#             behind_relation_dict = {
#                 'blue': ['red', 'green'],
#             }
#             left_relation_dict = {
#                 'green': ['red'],
#                 'cyan': ['purple'],
#             }
#             front_relation_dict = {
#                 'red': ['blue'],
#             }
#             right_relation_dict = {
#                 'red': ['green'],
#                 'purple': ['red', 'cyan'],
#                 'cyan': ['green'],
#             }
#         elif self.env_type == 'ordering':
#             target_satisfied_cnt = 4
#             behind_relation_dict = {}
#             left_relation_dict = {
#                 'red': ['blue'],
#                 'blue': ['green'],
#                 'green': ['purple'],
#                 'purple': ['cyan'],
#             }
#             front_relation_dict = {}
#             right_relation_dict = {}
#         else:
#             raise NotImplementedError
        
#         relation_list = [
#             copy.deepcopy(behind_relation_dict),
#             copy.deepcopy(left_relation_dict),
#             copy.deepcopy(front_relation_dict),
#             copy.deepcopy(right_relation_dict),
#         ]
#         self.target_relation_dict = dict(zip(dir_list, relation_list))
#         self.target_satisfied_cnt = target_satisfied_cnt
#         self.timestep = None
#         self.satisfied_cnt_list = None
        
#         self.low_env = LangGCPEnv(
#             maximum_episode_steps=50, action_type='perfect', obs_type='order_invariant', direct_obs=True, use_subset_instruction=True,
#             num_object=num_object,
#             language_model_type=self.language_model_type,
#             use_camera=use_camera,
#         )
        
#         device = f'cuda:{get_best_cuda()}'
#         model_path = models_dir.joinpath(f'{self.language_model_type}_0')
#         lm_kwargs = {
#             'device': device,
#             'model_path': model_path,
#             'hidden_dim': 128,
#             'output_dim': 14,
#             'policy_language_dim': 14,
#             'mode': 'low',
#             'is_kitchen': False,
#             'emb_dim': 768,
#             'epoch': 200,
#         }

#         kwargs = {
#             'features_extractor_kwargs': {
#                 'language_model_type': self.language_model_type,
#                 'language_model_kwargs': lm_kwargs,    
#             }
#         }
        
#         self.low_agent = LangGCPPPO.lang_load(
#             models_dir.joinpath('low').joinpath('policy_ag'),
#             device=device,
#             policy_kwargs=kwargs,
#             )
        
#         obs = self.reset()
#         self.observation_space = spaces.Box(
#             low=-10,
#             high=10,
#             shape=obs.shape,
#             dtype=np.float32,
#         )
#         if self.language_model_type == 'onehot':
#             cnt_list = [
#                 len(total_template_list),
#                 len(total_orientation_list),
#                 len(color_pair2color_idx),
#             ]
#             self.action_onehot_dim = int(np.prod(cnt_list))
#             self.action_space = spaces.Discrete(self.action_onehot_dim)
#         elif self.language_model_type in ['bert_cont', 'bert_onehot', 'bert_binary']:
#             self.action_onehot_dim = None
#             self.action_space = spaces.Box(low=-1, high=1, shape=(bert_cont_output_dim,))
#         elif 'policy' in self.language_model_type:
#             self.action_onehot_dim = 5
#             self.action_indicate_list = [2, 2, 2, 2, len(color_idx2color_pair)]
#             self.action_space = spaces.Discrete(np.prod(self.action_indicate_list))
#         elif self.language_model_type == 'human':
#             self.action_onehot_dim = None
#             self.action_indicate_list = [len(train_template_list), len(orientation_list), len(color_idx2color_pair)]
#             self.action_space = spaces.Discrete(np.prod(self.action_indicate_list))
#         elif self.language_model_type == 'no_lm':
#             self.action_onehot_dim = None
#             self.action_space = copy.deepcopy(self.low_env.action_space)
#         else:
#             raise NotImplementedError
        
#         if self.language_model_type != 'no_lm':
#             obs_no_goal = self.observation_space.sample()
#             goal_abs = self.action2goal_abs(self.action_space.sample())
#             self.low_agent.observation_space = spaces.Box(
#                 low=-10,
#                 high=10,
#                 shape=(obs_no_goal.shape[0] + goal_abs.shape[0],),
#                 dtype=np.float32,
#             )
#             self.low_agent.policy.observation_space = spaces.Box(
#                 low=-10,
#                 high=10,
#                 shape=(obs_no_goal.shape[0] + goal_abs.shape[0],),
#                 dtype=np.float32,
#             )

#     def update_kwargs_device(self, kwargs: dict, device: str) -> dict:
#         lm_kwargs = copy.deepcopy(kwargs['features_extractor_kwargs']['language_model_kwargs'])
        
#         lm_kwargs['device'] = device
        
#         if self.language_model_type == 'human':
#             mode = 'human'
#         else:
#             mode = 'high'
#         lm_kwargs['mode'] = mode
        
#         if self.language_model_type == 'onehot':
#             pass
#         elif self.language_model_type in ['bert_cont', 'bert_onehot', 'bert_binary', 'human']:
#             lm_kwargs['bert_kwargs']['device'] = device
#         elif self.language_model_type == 'policy':
#             pass
#         else:
#             raise NotImplementedError
        
#         kwargs['features_extractor_kwargs']['language_model_kwargs'] = lm_kwargs
        
#         return kwargs
    
#     def reset(self) -> np.ndarray:
#         self.low_env.reset()
        
#         self.timestep = 0
#         satisfied_cnt = self.compute_satisfied_relation()
#         self.satisfied_cnt_list = [satisfied_cnt]
        
#         obs = self.get_obs()

#         return obs

#     def get_obs(self) -> np.ndarray:
#         obs_no_goal_arr = self.low_env.get_obs_no_goal()
        
#         return obs_no_goal_arr

#     def compute_single_satisfied_relation(self, target_dir: int, relation_dict: dict) -> int:
#         assert target_dir in dir_list
        
#         single_satisfied_relation = 0
#         for dst_color, src_list in relation_dict.items():
#             for src_color in src_list:
#                 dir = self.low_env.compute_dir(src_color, dst_color)
#                 single_satisfied_relation += int(dir == target_dir)
        
#         return single_satisfied_relation

#     def compute_satisfied_relation(self) -> int:
#         satisfied_cnt = 0
#         for dir, relation_dict in self.target_relation_dict.items():
#             satisfied_cnt += self.compute_single_satisfied_relation(dir, relation_dict)
        
#         assert satisfied_cnt <= self.target_satisfied_cnt
        
#         return satisfied_cnt
    
#     def compute_reward(self) -> float:
#         if self.is_success():
#             reward = 0
#         else:
#             reward = -10
        
#         """
#         if self.is_fail():
#             reward = self.fail_reward
#         elif self.is_success():
#             reward = self.success_reward
#         else:
#             prev_satisfied_cnt = self.satisfied_cnt_list[-1]
#             curr_satisfied_cnt = self.compute_satisfied_relation()
#             self.satisfied_cnt_list.append(curr_satisfied_cnt)
            
#             reward = 1 * (curr_satisfied_cnt - prev_satisfied_cnt)
#         """
        
#         return reward

#     def is_success(self) -> bool:
#         satisfied_cnt = self.compute_satisfied_relation()
        
#         is_success = satisfied_cnt == self.target_satisfied_cnt
        
#         return is_success
    
#     def is_fail(self) -> bool:
#         is_fail = self.timestep >= self.maximum_episode_steps
        
#         return is_fail

#     def action2goal_abs(self, action) -> np.ndarray:
#         action = np.array(action)
#         if self.language_model_type == 'onehot':
#             onehot = np.zeros((action.size, self.action_onehot_dim))
#             onehot[np.arange(action.size), action] = 1
#             goal_abs = onehot.flatten()
#         elif self.language_model_type in ['bert_cont', 'bert_onehot', 'bert_binary']:
#             goal_abs = action
#         elif self.language_model_type == 'human':
#             tmp_action = action
#             goal_abs_list = []
#             for action_indicate in self.action_indicate_list:
#                 tmp_abs = tmp_action % action_indicate
#                 tmp_action = tmp_action // action_indicate
#                 goal_abs_list.append(tmp_abs)
#             goal_abs = np.array(goal_abs_list)
#         elif self.language_model_type == 'policy':
#             tmp_action = action
#             bool_abs_list = []
#             for idx in range(len(self.action_indicate_list) - 1):
#                 tmp_abs = tmp_action % self.action_indicate_list[idx]
#                 tmp_action = tmp_action // self.action_indicate_list[idx]
#                 bool_abs_list.append(tmp_abs)
#             bool_abs = np.array(bool_abs_list)
#             onehot_idx = np.array(color_idx2color_pair[tmp_action])
#             onehot_abs = np.zeros((onehot_idx.size, self.action_onehot_dim))
#             onehot_abs[np.arange(onehot_idx.size), onehot_idx] = 1
#             goal_abs = np.r_[bool_abs, onehot_abs.flatten()]
#         else:
#             raise NotImplementedError
        
#         return goal_abs

#     def action2language_goal(self, action) -> str:
#         raise NotImplementedError
        
#         tmp_action = action
#         goal_abs = self.action2goal_abs(action)

#         if self.language_model_type == 'onehot':
#             color_idx = tmp_action % len(color_pair2color_idx)
#             tmp_action = tmp_action // len(color_pair2color_idx)
#             orientation_idx = tmp_action % len(total_orientation_list)
#             tmp_action = tmp_action // len(total_orientation_list)
#             template_idx = tmp_action % len(total_template_list)

#             template = total_template_list[template_idx]
#             orientation = total_orientation_list[orientation_idx]
#             color_idx_arr = np.array(color_idx2color_pair[color_idx])
#             color_arr = self.low_env.color_arr[color_idx_arr]

#             language_goal = template.format(color_arr[0], orientation, color_arr[1])
#         elif self.language_model_type in ['bert_cont', 'bert_onehot', 'bert_binary']:
#             language_goal = ' '.join(np.array(goal_abs).astype(str))
#         elif self.language_model_type == 'policy':
#             language_goal = ' '.join(np.array(goal_abs).astype(str))
#         elif self.language_model_type == 'human':
#             template_idx, orientation_idx, color_idx = action

#             template = train_template_list[template_idx]
#             orientation = orientation_list[orientation_idx]
#             color_idx_arr = np.array(color_idx2color_pair[color_idx])
#             color_arr = self.low_env.color_arr[color_idx_arr]

#             language_goal = template.format(color_arr[0], orientation, color_arr[1])
#         else:
#             raise NotImplementedError

#         return language_goal

#     def step(self, action):
#         if self.language_model_type != 'no_lm':
#             goal_abs = self.action2goal_abs(action)

#             self.low_env.high_step_setup(goal_abs)

#             low_obs, low_reward, low_done, low_info = self.low_env.high_step(self.low_agent, self.n_step)
#         else:
#             low_obs, low_reward, low_done, low_info = self.low_env.step(action)

#         self.timestep += 1

#         obs = self.get_obs()
#         reward = self.compute_reward()
        
#         is_fail = self.is_fail()
#         is_success = self.is_success()
#         done = is_fail or is_success
        
#         custom_info = {}
#         if done:
#             custom_info['satisfied_cnt_min'] = np.min(self.satisfied_cnt_list)
#             custom_info['satisfied_cnt_mean'] = np.mean(self.satisfied_cnt_list)
#             custom_info['satisfied_cnt_max'] = np.max(self.satisfied_cnt_list)
#         info = {
#             'is_fail': is_fail,
#             'is_success': is_success,
#             'custom_info': custom_info,
#         }

#         return obs, reward, done, info

#     def render(self, mode="human"):
#         return self.low_env.render(mode)


# class EvalLangEnv(LangGCPEnv):
#     def _reward(self):
#         prev_dist = self.prev_dist
#         curr_dist = self.compute_dist()
#         bias = self.compute_bias()
#         bias_max = np.max(bias)
#         reward_info = self.reward_info_template.copy()

#         if self.is_fail():
#             reward = 0
#             reward_info['result'] = self.fail_reward
#         elif self.is_success():
#             reward = 1
#             reward_info['result'] = self.success_reward
#         else:
#             near_reward = (prev_dist - curr_dist) * self.dist_scale
#             still_reward = -bias_max * self.dist_scale * 0.1

#             reward_info['near'] = near_reward
#             reward_info['still'] = still_reward
            
#             reward = 0

#         self.reward_info_list.append(reward_info)
#         self.bias_list.append(bias_max.item())
#         self.dist_list.append(self.prev_dist)

#         return reward


# class LangComplexEnv(LangGCPEnv):
#     def __init__(self, maximum_episode_steps=100, xml_path=None, metadata_path=None, template_path=None, num_object=5, agent_type='pm', random_start=False, description_num=15, action_type='continuous', obs_type='direct', use_movement_bonus=False, direct_obs=False, reward_scale=1, frame_skip=20, shape_val=0.25, min_move_dist=0.05, resolution=64, use_synonyms=False, min_change_th=0.26, use_polar=False, use_subset_instruction=False, systematic_generalization=False, suppress_other_movement=False, top_down_view=False, variable_scene_content=False, fail_dist=0.2, train_template_cnt=9, language_model_type='onehot', mode='train', use_camera=False):
#         self.second_template_idx = None
#         self.second_orientation_idx = None
#         self.second_color_idx = None
        
#         self.second_total_template_arr = np.array(second_total_template_list)
#         self.second_train_template_arr = np.array(second_train_template_list)[:train_template_cnt]
#         self.second_error_template_arr = np.array(second_error_template_list)
#         self.second_test_template_arr = np.array(second_test_template_list)
        
#         self.second_total_orientation_arr = np.array(second_total_orientation_list)
#         self.second_orientation_arr = np.array(second_orientation_list)
        
#         if mode == 'train':
#             self.reset_second_template_arr = self.second_train_template_arr.copy()
#             self.reset_second_orientation_arr = self.second_orientation_arr.copy()
#         elif mode == 'test':
#             self.reset_second_template_arr = self.second_test_template_arr.copy()
#             self.reset_second_orientation_arr = self.second_orientation_arr.copy()
#         elif mode == 'error':
#             self.reset_second_template_arr = self.second_test_template_arr.copy()
#             self.reset_second_orientation_arr = self.second_total_orientation_arr.copy()
#         else:
#             raise NotImplementedError

#         self.second_dist_list = None
#         self.second_dir = None
#         self.second_target_xy = None
        
#         super().__init__(maximum_episode_steps, xml_path, metadata_path, template_path, num_object, agent_type, random_start, description_num, action_type, obs_type, use_movement_bonus, direct_obs, reward_scale, frame_skip, shape_val, min_move_dist, resolution, use_synonyms, min_change_th, use_polar, use_subset_instruction, systematic_generalization, suppress_other_movement, top_down_view, variable_scene_content, fail_dist, train_template_cnt, language_model_type, mode, use_camera)

#     def language_goal2self_goal(self,
#                                 recons_color_idx_arr: np.ndarray,
#                                 recons_orientation_idx: int,
#                                 recons_second_color_idx: int = None,
#                                 recons_second_orientation_idx: int = None,
#                                 ):
#         self.goal_arr = np.zeros(self.num_object + 1)
        
#         assert self.goal_count == 2
#         assert recons_color_idx_arr.size == self.goal_count
#         goal_src_idx, goal_dst_idx = recons_color_idx_arr
#         self.goal_arr[goal_src_idx] = self.goal_src
#         self.goal_arr[goal_dst_idx] = self.goal_dst
#         self.goal_arr[-1] = recons_orientation_idx
        
#         self.dir = recons_orientation_idx
#         self.delta_xy = self.update_delta_xy(dir=self.dir)
        
#         if recons_second_color_idx is not None and recons_second_orientation_idx is not None:
#             self.second_dir = recons_second_orientation_idx
#             second_curr_xy = self.get_obs_no_goal().reshape(self.num_object, -1)[recons_second_color_idx]
#             second_delta_xy = self.update_delta_xy(dir=self.second_dir, is_second=True)
#             self.second_target_xy = second_curr_xy + second_delta_xy

#     def reset_mdp_utils(self, dir=None):
#         new_template_idx = np.random.choice(np.arange(self.reset_template_arr.size))
#         new_template = self.reset_template_arr[new_template_idx]
#         new_second_template_idx = new_template_idx
#         new_second_template = self.reset_second_template_arr[new_second_template_idx]
        
#         new_orientation_idx = np.random.choice(np.arange(self.reset_orientation_arr.size))
#         new_orientation = self.reset_orientation_arr[new_orientation_idx]
#         new_second_orientation_idx = np.random.choice(np.arange(self.reset_second_orientation_arr.size))
#         new_second_orientation = self.reset_second_orientation_arr[new_second_orientation_idx]

#         new_color_idx_arr = np.random.choice(np.arange(self.color_arr.size), self.goal_count, replace=False)
        
#         second_color_list = color_list.copy()
#         second_color_list.pop(new_color_idx_arr[0])
#         new_second_color = np.random.choice(second_color_list)
#         new_second_color_idx = color_list.index(new_second_color)

#         new_color_arr = self.color_arr[new_color_idx_arr]
        
#         first_goal = new_template.format(new_color_arr[0], new_orientation, new_color_arr[1])
#         second_goal = new_second_template.format(new_second_color, new_second_orientation)
#         language_goal = first_goal + second_goal
#         self.language_goal = language_goal
        
#         onehot_template_idx = np.where(self.total_template_arr == new_template)[0].item()
#         onehot_orientation_idx = np.where(self.total_orientation_arr == new_orientation)[0].item()
        
#         onehot_second_template_idx = np.where(self.second_total_template_arr == new_second_template)[0].item()
#         onehot_second_orientation_idx = np.where(self.second_total_orientation_arr == new_second_orientation)[0][0].item()
        
#         self.template_idx = onehot_template_idx
#         self.orientation_idx = onehot_orientation_idx
#         self.color_idx = color_pair2color_idx[(new_color_idx_arr[0], new_color_idx_arr[1])]
        
#         self.second_template_idx = onehot_second_template_idx
#         self.second_orientation_idx = onehot_second_orientation_idx
#         self.second_color_idx = new_second_color_idx
        
#         obs = super().reset_mdp_utils(dir)
        
#         recons_color_idx_arr = new_color_idx_arr
#         recons_second_color_idx = new_second_color_idx
        
#         if self.mode in ['train', 'test']:
#             recons_orientation_idx = new_orientation_idx
#             recons_second_orientation_idx = new_second_orientation_idx
#         elif self.mode in ['error']:
#             recons_orientation_idx = new_orientation_idx // 3
#             recons_second_orientation_idx = new_second_orientation_idx // 3
#         else:
#             raise NotImplementedError
        
#         self.language_goal2self_goal(
#             recons_color_idx_arr=recons_color_idx_arr,
#             recons_orientation_idx=recons_orientation_idx,
#             recons_second_color_idx=recons_second_color_idx,
#             recons_second_orientation_idx=recons_second_orientation_idx,
#         )

#         self.goal_abs = None
#         self.second_dist_list = [self.compute_second_dist()]

#         return obs

#     def compute_second_dist(self) -> float:
#         shaped_obs_no_goal = self.get_obs_no_goal().reshape(self.num_object, -1)
        
#         curr_xy = shaped_obs_no_goal[self.second_color_idx].copy()
#         target_xy = self.second_target_xy.copy()
        
#         dist = np.linalg.norm(target_xy - curr_xy)
        
#         return dist

#     def _reward(self):
#         prev_dist = self.prev_dist
#         curr_dist = self.compute_dist()
        
#         prev_second_dist = self.second_dist_list[-1]
#         curr_second_dist = self.compute_second_dist()
        
#         bias = self.compute_bias()
#         bias_max = np.max(bias)
#         reward_info = self.reward_info_template.copy()

#         if self.is_fail():
#             reward = self.fail_reward
#             reward_info['result'] = self.fail_reward
#         elif self.is_success():
#             reward = self.success_reward
#             reward_info['result'] = self.success_reward
#         else:
#             near_reward = (prev_dist - curr_dist) * self.dist_scale
            
#             second_near_reward = (prev_second_dist - curr_second_dist) * self.dist_scale
            
#             still_reward = -bias_max * self.dist_scale * 0.1

#             reward = near_reward + second_near_reward + still_reward
            
#             reward_info['near'] = near_reward
#             reward_info['second_near'] = second_near_reward
#             reward_info['still'] = still_reward

#         self.reward_info_list.append(reward_info)
#         self.bias_list.append(bias_max.item())
#         self.dist_list.append(self.prev_dist)
#         self.second_dist_list.append(curr_second_dist)

#         return reward

#     def is_success(self):
#         goal_src_xy = self.get_goal_xy(self.goal_src).copy()
#         goal_dst_xy = self.get_goal_xy(self.goal_dst).copy()
#         curr_angle = self.compute_angle(goal_src_xy)

#         curr_dist = self.compute_dist(goal_dst_xy)
#         curr_second_dist = self.compute_second_dist()

#         angle_rg = dir2angle_rg[self.dir]
#         if angle_rg[1] < angle_rg[0]:
#             angle_flag = (angle_rg[0] <= curr_angle < 360) or (0 <= curr_angle < angle_rg[1])
#         else:
#             angle_flag = angle_rg[0] <= curr_angle < angle_rg[1]
#         dist_flag = curr_dist < self.success_dist_threshold and curr_second_dist < self.success_dist_threshold

#         is_success = angle_flag and dist_flag
        
#         return is_success

#     def get_obs(self) -> np.ndarray:
#         obs_no_goal_arr = self.get_obs_no_goal()
        
#         if self.language_goal is None:
#             language_goal_arr = np.zeros(6)
#         else:
#             language_goal_arr = np.array([
#                 self.template_idx, 
#                 self.orientation_idx, 
#                 self.color_idx,
#                 self.second_template_idx, 
#                 self.second_orientation_idx, 
#                 self.second_color_idx,
#                 ])

#         obs = np.r_[obs_no_goal_arr, language_goal_arr]

#         return obs
    