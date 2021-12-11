import os
import argparse
import pprint

import numpy as np

# import pybulletgym
import gym

import torch
from tianshou.policy import TD3Policy
from tianshou.exploration import GaussianNoise
from utils import *
from robot import AntRobot
from net import Actor, Critic


def communicationBetweenNNandAntRobot():
    # Initialize the real Ant robot
    ant_robot = AntRobot()
    print("Initialized")
    # Get the initial observation
    obs = ant_robot.reset()
    print("Get observation")
    # Load the pretrained agent with NN
    agent = get_trained_agent()
    print("Get get_trained_agent")
    done = False

    for i in range(6000):
        # obs should change to size: 1 x 16
        totalenergy = obs[-1]
        obs = obs[0:16]
        # obs = np.hstack((np.clip(obs[-1], 0.01, 0.3), obs[:-1]))
        obs = observationRegularization(obs)
        obs = np.array(obs).reshape(1, 16)
        # obs = obs + (np.random.rand(16)-0.5)*0.2
        # print("The observation is:")
        # print(obs)

        # the output size of action : 1 x 8
        action = agent.forward_numpy(obs).detach().numpy()

        # action = agent.forward(obs)
        # action = ( action + np.random.rand(1, 8) * 0.3 ) * (150 / np.pi)

        # action should change to size: 8 x 1
        action = np.array(action).reshape(8,)
        # original: hip4 ank4 hip1 ank1 hip2 ank2 hip3 ank3 
        # change to: hip1 hip2 hip3 hip4 ank1 ank2 ank3 ank4
        action = [action[2], action[4], action[6], action[0],\
                  action[3], action[5], action[7], action[1]]
        # action = [action[6], action[4], action[6], action[4],\  # symmetric
        #           action[7], action[5], action[7], action[5]]
        # print("The action is:")
        # print(action)
        obs, done = ant_robot.step(action)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor-lr', type=float,        
 default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_known_args()[0]
    return args


def get_trained_agent(args=get_args()):

    args.state_shape = (16,)
    args.action_shape = (8,)
    args.max_action = 1
    
    # model
    actor = Actor(
        args.layer_num,
        args.state_shape, args.action_shape,
        args.max_action, args.device).to(args.device)
    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=args.actor_lr)

    critic1 = Critic(
        args.layer_num,
        args.state_shape, args.action_shape,
        args.device).to(args.device)
    critic1_optim = torch.optim.Adam(
        critic1.parameters(),
        lr=args.critic_lr)

    critic2 = Critic(
        args.layer_num,
        args.state_shape, args.action_shape,
        args.device).to(args.device)
    critic2_optim = torch.optim.Adam(
        critic2.parameters(),
        lr=args.critic_lr)

    policy = TD3Policy(
        actor, actor_optim,
        critic1, critic1_optim,
        critic2, critic2_optim,
        [-1, 1],
        args.tau, args.gamma,
        args.policy_noise,
        args.update_actor_freq,
        args.noise_clip,
        reward_normalization=True, ignore_done=True)

    policy.eval()

    ####################################################
    # Load the trained model parameters
    load = True
    if load:
        # logdir = 'log/AntPyBullet-v0/four_state/td3'
        # model_path = logdir + '/' + 'policy_500.pth'
        logdir = 'log/td3_16observation_cs285Ant2'
        model_path = logdir + '/' + 'policy.pth'
        if os.path.exists(model_path):
            policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    ####################################################

    return policy


if __name__ == "__main__":
    communicationBetweenNNandAntRobot()
