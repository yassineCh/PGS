import numpy as np
import torch
import pickle
from tqdm import tqdm

clip_action = 10

def get_grads(x, f, device="cuda"):
    x_tensor = torch.from_numpy(x).to(device)
    x_tensor.requires_grad_()
    y = f(x_tensor)
    grad_f_x = torch.autograd.grad(y.sum(), x_tensor)[0].cpu().detach().numpy() 
    
    return grad_f_x

def link(x1, grads):
    ## link observations through gradients
    l =  x1[1:] - x1[:-1]
    action = l / grads
    action = np.nan_to_num(action)/clip_action
    action[np.abs(action) > 1] = 0   
    
    return action

def generate_trajectory(x, y, f, length=50, device="cuda"):
    # generate one trajectory with specified length
    idx_0 = np.random.choice(np.arange(x.shape[0]), length+1, replace=False)
    x_0 =  x[idx_0] 
    grads = get_grads(x_0[:-1], f, device)
    reward = y[idx_0[1:]] - y[idx_0[:-1]]
    traj = {'observations':[],
            "next_observations":[],
            'actions':[],
            'rewards':[],
            'dones':[],
            }
    action = link(x_0, grads)
    #save data
    traj['observations'] = x_0[:-1]
    traj['next_observations'] = x_0[1:]
    traj['actions'] = action
    traj['rewards'] = reward
    #episode length is T=50
    traj['dones'] = np.array([0]*49 + [1])
    
    return traj

def generate_offline_dataset(x, y, surrogate, dataset_name, size=20000, length=50, device="cuda"):
    # create a dataset of observation for offline RL
    dataset = {'observations':[],
            'next_observations':[],
            'actions':[],
            'rewards':[],
            'dones':[],}

    for j in tqdm(range(size)):
        traj = generate_trajectory(x, y, surrogate, length, device=device)
        dataset['observations'].append(traj['observations'])
        dataset['next_observations'].append(traj['next_observations'])
        dataset['actions'].append(traj['actions'])
        dataset['rewards'].append(traj['rewards'])
        dataset['dones'].append(traj['dones'])

    print("saving dataset ....")
    observations = np.array(dataset["observations"])
    next_observations = np.array(dataset["next_observations"])
    actions = np.array(dataset["actions"])
    rewards = np.array(dataset["rewards"])
    dones = np.array(dataset["dones"])

    observations = observations.reshape((-1, observations.shape[-1]))
    next_observations = next_observations.reshape((-1, next_observations.shape[-1]))
    actions = actions.reshape((-1, actions.shape[-1]))
    rewards = rewards.reshape((-1, rewards.shape[-1]))
    dones = dones.reshape((-1, 1))
    
    traj['observations'] = observations
    traj['next_observations'] = next_observations
    traj['actions'] = actions
    traj['rewards'] = rewards
    traj['dones'] = dones

    with open('traj_'+dataset_name+'.pickle', 'wb') as handle:
        pickle.dump(traj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return traj


