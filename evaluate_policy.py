import numpy as np
import torch
import tensorflow as tf
from generate_trajectories import get_grads
 
def load_top_observations(x, y, top_k):
    top_indices = tf.math.top_k(y[:, 0], k=top_k)[1]
    top_indices = top_indices.numpy()
    top_vals = y[top_indices]
    top_observations = x[top_indices, :]

    return top_observations, top_vals

def step(observation, action, surrogate, scale=10, device="cuda"):
    grads = get_grads(observation, surrogate, device=device)
    next_observation = observation + action * scale * grads 
    
    return next_observation

def generate_designs(policy, surrogate, start_observations, scale=10,
            max_traj_length=50, deterministic=True, device="cuda"):
    
    ## sample trajectories following policy
    for _ in range(max_traj_length):
        obs_tensor = np.array(start_observations)
        obs_tensor = torch.from_numpy(obs_tensor).to(device)

        action = policy(obs_tensor, deterministic=deterministic)#[0, :]
        action = action[0].cpu().detach().numpy()

        next_observation = step(start_observations, action, surrogate, scale=scale, vec=True)
        start_observations = next_observation

    return next_observation

def oracle_evaluate_designs(task, task_dataset, x, normalize_ys=True, discrete=False):
    if discrete:
        x = np.array(x).astype(np.int64)
    
    y_min, y_max = task_dataset.y.min(), task_dataset.y.max()
    y = task.predict(x)
    if normalize_ys:
        y = task.denormalize_y(y)
    max_y = (np.max(y)-y_min)/(y_max-y_min)
    
    return max_y


        