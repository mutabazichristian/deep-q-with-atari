import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import *
import numpy as np

# loading the environment
# gym.register_envs(ale_py)

# env = gym.make("ALE/Breakout-v5", render_mode='human')
# use render mode to get visual, But it will slow down game play
# env = gym.make('ALE/Breakout-v5',render_mode='human')
# Custom wrapper to make ALE/Breakout-v5 work like SB3
def make_custom_atari_env():
    env = gym.make("ALE/Breakout-v5", render_mode='human')

    # Apply wrappers similar to SB3's default Atari pipeline
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)

    return env

# Wrap into VecEnv and stack 4 frames
env = DummyVecEnv([make_custom_atari_env])
env = VecFrameStack(env, n_stack=4)

# loading the agent
model = DQN.load("dqn_model")

obs = env.reset()
total_reward = 0

for step in range(1000):
    # now we use age to predict action
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward

    if step % 100 == 0:
        print(f"Step {step}, Reward: {total_reward}")

    if done[0]:
        print(f"Episode ended at step {step}, Total reward: {total_reward}")
        obs = env.reset()
        total_reward = 0

env.close()
