import torch
from offlinerl.utils.exp import select_free_cuda

task = "Simglucose"
task_data_type = "medium"
task_train_num = 99

seed = 42

device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

# model save path
policy_bc_path = None
policy_bc_save_path = None
dynamics_path = None
dynamics_save_path = None

# transition model train
transition_init_num = 7
transition_select_num = 5
val_ratio = 0.2
max_epochs_since_update = 5
transition_max_epochs = None

# trick config
trainsition_clip = True
normalize_obs = False
transition_scaler = True
policy_scaler = True

# transition config
transition_batch_size = 256
transition_lr = 1e-3  # 3e-4
logvar_loss_coef = 0.01  # 1e-3
dynamics_hidden_dims = [200, 200, 200, 200]
dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]

# alpha config
learnable_alpha = True
alpha_lr = 1e-4
alpha = 0.2

# train config
horizon = 5
real_data_ratio = 0.5
max_epoch = 2000
steps_per_epoch = 1000
rollout_freq = 250
rollout_batch_size = 5e+4

# policy config
hidden_dims = [256, 256]
policy_batch_size = 256
actor_lr = 1e-4

# critic config
critic_lr = 3e-4
discount = 0.99
soft_target_tau = 5e-3
target_entropy = None

# others
val_frequency = 10
eval_episodes = 10
model_retain_epochs = 5

# rambo config
policy_bc_epoch = 50
policy_bc_batch_size = 256
policy_bc_lr = 1e-4

transition_adv_lr = 3e-4
dynamics_update_freq = 1000
adv_train_steps = 1000
adv_rollout_batch_size = 256
adv_rollout_length = 5
include_ent_in_adv = False
adv_weight = 3e-4

#tune
params_tune = {
    "real_data_ratio" : {"type" : "discrete", "value": [0.05, 0.1, 0.2]},
    "horizon" : {"type" : "discrete", "value": [1, 2, 5]},
    "adv_weight" : {"type" : "discrete", "value": [0, 3e-4]},
}

#tune
grid_tune = {
    "horizon" : [1, 5],
    "transition_adv_lr" : [1e-3, 3e-4],
    "adv_weight" : [0, 1e-3, 3e-4],
}