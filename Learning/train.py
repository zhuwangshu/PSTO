import os
import argparse
import pprint

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import gym
import pybulletgym

from tianshou.policy import TD3Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import VectorEnv, SubprocVectorEnv
from randomNoise import GaussianNoise

from net import Actor, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='AntPyBulletEnv-v0') #AntPyBulletEnv-v0  HalfCheetahPyBulletEnv-v0 HalfCheetahMuJoCoEnv-v0
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)

    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=200)
    parser.add_argument('--layer-num', type=int, default=3)
    

    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=50)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def train(args=get_args()):

    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    
    # train_envs = gym.make(args.task)
    # test_envs = gym.make(args.task)
    train_envs = VectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
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
        args.tau, args.gamma,
        args.exploration_noise, args.policy_noise,
        args.update_actor_freq,
        args.noise_clip,
        [env.action_space.low[0], env.action_space.high[0]],
        reward_normalization=True, ignore_done=True)

    # policy = TD3Policy(
    #     actor, actor_optim,
    #     critic1, critic1_optim,
    #     critic2, critic2_optim,
    #     args.tau, args.gamma,
    #     GaussianNoise(sigma=0.1), args.policy_noise, #GaussianNoise(sigma=0.1), args.policy_noise,
    #     args.update_actor_freq,
    #     args.noise_clip,
    #     [env.action_space.low[0], env.action_space.high[0]],
    #     reward_normalization=True, ignore_done=True)

    # log
    log_path = os.path.join(args.logdir, args.task, 'td3_285_ant_motorCost-1.0')
    writer = SummaryWriter(log_path)

    ####################################################
    # Load the trained model parameters
    load = False
    if load:
        model_path = log_path + '/' + 'policy.pth'
        if os.path.exists(model_path):
            print("Model loaded from the path: " + model_path)
            policy.load_state_dict(torch.load(model_path))
    ####################################################

    # collector

    train_collector = Collector(policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)

    # train_collector.collect(n_step=args.buffer_size)
    


    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    # args.epoch: number of time (epoch) for training 
    # args.step_per_epoch: number of step for updating policy network in one epoch
    # args.collect_per_step: number of data points we collect before each policy updating 

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        args.epoch, args.step_per_epoch, args.collect_per_step,
        args.test_num,
        args.batch_size,
        save_fn=save_fn, stop_fn=stop_fn,
        writer=writer, task=args.task)

    # assert stop_fn(result['best_reward'])

    train_collector.close()
    test_collector.close()

    show_training_performance = True
    if show_training_performance:
        pprint.pprint(result)
        env = gym.make(args.task)
        env.render()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    train()


