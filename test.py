import ale_py
import gymnasium as gym

gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', render_mode='human')

obs, info = env.reset()
total_reward = 0

for step in range(1000):
   action = env.action_space.sample()  # Random action
   obs, reward, terminated, truncated, info = env.step(action)
   total_reward += reward
   
   if step % 100 == 0:
       print(f"Step {step}, Reward: {total_reward}")
   
   if terminated or truncated:
       print(f"Episode ended at step {step}, Total reward: {total_reward}")
       obs, info = env.reset()
       total_reward = 0

env.close()
