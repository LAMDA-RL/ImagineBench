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
max_epochs_since_update = 10
transition_max_epochs = None

# trick config
normalize_obs = False
transition_scaler = True

# transition config
transition_batch_size = 256
transition_lr = 1e-3
logvar_loss_coef = 0.01
dynamics_hidden_dims = [200, 200, 200, 200]
dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]

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
    "transition_scaler" : [True, False],
    "transition_lr" : [1e-3, 3e-4],
    "logvar_loss_coef" : [0.01, 1e-3],
}
