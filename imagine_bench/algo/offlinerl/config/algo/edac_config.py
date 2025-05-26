import torch
from offlinerl.utils.exp import select_free_cuda
# from datetime import datetime

task = "Hopper-v3"
device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'


# parser.add_argument("--algo-name", type=str, default="edac")
    # parser.add_argument("--task", type=str, default="SafetyHalfCheetah")
obs_shape = None
act_shape = None

seed = 42
actor_lr=1e-4
critic_lr=3e-4
task_train_num = 99
task_data_type = 'high'
# hidden_dims=[256, 256, 256]
hidden_layer_size = 256
layer_num = 2
gamma=0.99
tau=0.005
alpha=0.2
auto_alpha=True

target_entropy = None
alpha_lr =1e-4
num_critics = 50

max_q_backup = False
deterministic_backup=False

eta=1.0
normalize_reward=False

epoch=3000
step_per_epoch=1000

eval_episodes=100
batch_size=256

#tune
params_tune = {
    "num_critics" : {"type" : "discrete", "value":[10,50]},
    "eta" : {"type" : "discrete", "value":[1, 5]},
}

grid_tune = {
    "num_critics" : [10, 50],
    "eta" : [1, 5],
}


# task_data_type = "low"
# task_train_num = 99

# seed = 42

# device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
# obs_shape = None
# act_shape = None
# max_action = None

# max_epoch = 300
# steps_per_epoch = 1000
# policy_bc_steps = 40000

# batch_size = 256
# hidden_layer_size = 256
# layer_num = 2
# actor_lr=1E-4
# critic_lr=3E-4
# reward_scale=1
# use_automatic_entropy_tuning=True
# target_entropy = None
# discount = 0.99
# soft_target_tau=5e-3

# # min Q
# explore=1.0
# temp=1.0
# min_q_version=3
# min_q_weight=5.0

# # lagrange
# with_lagrange=False
# lagrange_thresh=2.0

# # extra params
# num_random=10
# type_q_backup= "min"
# q_backup_lmbda = 0.75
# deterministic_backup=False

# discrete = False

#tune
# params_tune = {
#     "actor_lr" : {"type" : "discrete", "value":[1e-4, 3e-4]},
#     "min_q_version" : {"type" : "discrete", "value":[2, 3]},
#     "min_q_weight" : {"type": "discrete", "value":[5, 10]},
#     "lagrange_thresh" : {"type": "discrete", "value":[-1, 2, 5, 10]},
#     "type_q_backup" : {"type": "discrete", "value":["max", "none"]},
# }

# #tune
# grid_tune = {
#     #"actor_lr" : [1e-4, 3e-4],
#     "min_q_version" : [2, 3],
#     "min_q_weight" : [5, 10],
#     "lagrange_thresh" : [-1, 2, 5, 10],
#     # "type_q_backup" : ["min", "none"],
# }
