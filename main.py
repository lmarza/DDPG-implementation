import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agent import Agent
from utils import *

env = NormalizedEnv(gym.make("Pendulum-v0"))

agent = Agent(env)
batch_size = 128
agent.train(max_episode=100, max_step=1000, batch_size=batch_size, env=env)
