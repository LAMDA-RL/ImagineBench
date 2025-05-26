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


actor_features = 256
actor_layers = 2
value_features = 256
value_layers = 2

alpha = 2.5
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2


batch_size = 256
steps_per_epoch = 1000
max_epoch = 100


actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
discount = 0.99
soft_target_tau = 5e-3

num_sampled_actions = 10
eval_episodes = 100

#tune
grid_tune = {
    "alpha" : [0.05, 0.1, 0.2],
    "policy_noise" : [0.5, 1.5, 2.5],
}
