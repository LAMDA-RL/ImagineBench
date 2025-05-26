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

vae_features = 750
vae_layers = 2
actor_features = 400
actor_layers = 2
value_features = 400
value_layers = 2
lam = 0.95

alpha = 0.2
auto_alpha = True
target_entropy = None

batch_size = 256
steps_per_epoch = 1000
max_epoch = 1000

vae_lr = 1e-3
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
gamma = 0.99
soft_target_tau = 5e-3

num_sampled_actions = 10
eval_episodes = 100

#tune
params_tune = {
    "lam" : {"type" : "continuous", "value": [0.3, 0.95]},
}

#tune
grid_tune = {
    "lam" : [0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "auto_alpha" : [True, False],
}
