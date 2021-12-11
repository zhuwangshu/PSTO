import gym
import pybulletgym

import numpy as np
import time


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def test_PybulletCustomAntEnv():
    env = gym.make('HalfCheetahPyBulletEnv-v0')
    agent = RandomAgent(env.action_space)
    env.render()

    obs = env.reset()
    reward = 0
    done = False
    renderStep = 0.1
    while True:
        action = agent.act(obs, reward, done)
        # action = [30,0,0,0,0,0,0,0]
        obs, reward, done, _ = env.step(action)
        print("obs",obs)
        print("action",action)
        env.render()
        # time.sleep(renderStep)



if __name__ == '__main__':
    test_PybulletCustomAntEnv()

