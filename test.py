import random
from tqdm import tqdm
import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from common import utils
from common.model import Policy
from common.storage import RolloutStorage
from common.arguments import get_args
from gridworld.env2 import twoColorsEnv
from gridworld.utils import storage, setup_seed
from gridworld.plot import plot

from stable_baselines3.common.monitor import Monitor


# Not support parallel sampling since environment is self-defined

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    splits = args.log_dir.split('/')
    sub_dir = splits[-1] if splits[-1] != '' else splits[-2]
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    env = twoColorsEnv()
    env = Monitor(env, args.log_dir)

    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'trpo':
        agent = algo.TRPO(
            actor_critic,
            max_kl = args.max_kl,
            damping = args.damping,
            l2_reg = args.l2_reg,
        )

    
    obs = env.reset()
    obs = torch.FloatTensor(obs).unsqueeze(0)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)


    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    for j in tqdm(range(num_updates)):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, entropy, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = env.step(action)
                                
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in [done]])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in [infos]])

            if done:
                obs = env.reset()
                         
            obs    = torch.FloatTensor(obs).unsqueeze(0)
            reward = torch.FloatTensor([reward]).unsqueeze(0)
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, entropy, masks, bad_masks)

        with torch.no_grad():
            next_value, next_entropy = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1])
            rollouts.entropies[-1] = next_entropy

        
        rollouts.augment_rewards(args.gamma, args.alpha, args.augment_type)
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
            

if __name__ == "__main__":
    main()


