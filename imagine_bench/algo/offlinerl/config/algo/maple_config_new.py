import torch
from offlinerl.utils.exp import select_free_cuda

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42

device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

# transition model train
transition_init_num = 20
transition_select_num = 14
val_ratio = 0.2
max_epochs_since_update = 5
transition_max_epochs = None

# trick config
trainsition_clip = False
normalize_obs = False  # should set to False
transition_scaler = True

# transition config
transition_batch_size = 256
transition_lr = 1e-3
logvar_loss_coef = 0.01
dynamics_hidden_dims = [200, 200, 200, 200]
dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]

# new parameters based on mopo
lstm_hidden_unit = 128
Guassain_hidden_sizes = (256,256)
value_hidden_sizes=(256,256)
hidden_sizes=(16,)
model_pool_size = 250000
rollout_batch_size = 50000
handle_per_round = 400
out_train_epoch = 1000
in_train_epoch = 1000 

train_batch_size = 256              # train policy num of trajectories

number_runs_eval = 40            # evaluation epochs in mujoco 

#-------------
dynamics_path = None
dynamics_save_path = None
only_dynamics = False

hidden_layer_size = 256
hidden_layers = 2

real_data_ratio = 0.05

policy_batch_size = 256
data_collection_per_epoch = 50e3
steps_per_epoch = 1000
max_epoch = 1000

eval_episodes = 100

# alpha config
learnable_alpha = True
alpha_lr = 1e-4
alpha = 0.2
target_entropy = None

uncertainty_mode = 'aleatoric'
actor_lr = 3e-4
critic_lr = 3e-4
discount = 0.99
soft_target_tau = 5e-3

horizon = 10
penalty_coef = 0.25

penalty_clip = 20

#tune
params_tune = {
    "buffer_size" : {"type" : "discrete", "value": [1e6, 2e6]},
    "real_data_ratio" : {"type" : "discrete", "value": [0.05, 0.1, 0.2]},
    "horzion" : {"type" : "discrete", "value": [1, 2, 5]},
    "lam" : {"type" : "continuous", "value": [0.1, 10]},
    "learnable_alpha" : {"type" : "discrete", "value": [True, False]},
}

#tune
grid_tune = {
    "horizon" : [1, 5],
    "lam" : [0.5, 1, 2, 5],
    "uncertainty_mode" : ['aleatoric', 'disagreement'],
}
