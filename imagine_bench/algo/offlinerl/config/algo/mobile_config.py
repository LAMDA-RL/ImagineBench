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

# model save path
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
policy_scaler = False

# transition config
transition_batch_size = 256
transition_lr = 1e-3
logvar_loss_coef = 0.01
dynamics_hidden_dims = [200, 200, 200, 200]
dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]

# alpha config
learnable_alpha = True
alpha_lr = 1e-4
alpha = 0.2

# train config
horizon = 5
real_data_ratio = 0.05
max_epoch = 3000
steps_per_epoch = 1000
rollout_freq = 1000
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

# mobile config
num_q_ensemble = 2
penalty_coef = 3.5
num_samples = 10

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
    "penalty_coef" : [0.5, 1.5, 2.5, 3.5],
    "real_data_ratio" :[0.05], 
}
