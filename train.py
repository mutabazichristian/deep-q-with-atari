import gymnasium as gym
import ale_py
import os
import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback

"""
CNN and MLP Policies comparison

CNN is better at visual processing like 'seeing' the bricks on the wall and locating the character/agent as well as the ball. 

Expected Features would be looking at the edges, location of the ball and direction, and the agents location.

MLP would have problems because of the large input size and wouldn't be able to comprehend spatial features like CNNs would do. 

let's test this and see
"""


class DeepQ:
    def __init__(
        self, env_name="ALE/Breakout-v5", total_timesteps=30000, seed=42
    ):
        """init"""
        self.env_name = env_name
        self.total_timesteps = total_timesteps
        self.seed = seed
        self.results = {}

        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        gym.register_envs(ale_py)

    def create_env(self):
        "create atari env"
        env = make_atari_env(self.env_name, n_envs=1, seed=self.seed)
        env = VecFrameStack(env, n_stack=4)
        return env

    def train_model(self, policy_type, model_name):
        """train deep agent with specic"""
        print(f"\n=== Training DQN with {policy_type} ===")

        # Create env
        env = self.create_env()

        # Create model
        model = DQN(
            policy=policy_type,
            env=env,
            learning_rate=5e-5,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=16,
            tau=1.0,
            gamma=0.92,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            max_grad_norm=10,
            verbose=1,
            seed=self.seed,
        )

        # Set up evaluation callback
        eval_env = self.create_env()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/{model_name}",
            log_path=f"./logs/{model_name}",
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
        )

        # Train the model
        start_time = time.time()
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=eval_callback,
            log_interval=100,
        )
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")

        # Save final model
        model.save(f"{model_name}_final")

        env.close()
        eval_env.close()

        # Save results
        self.results[policy_type] = {
            "model": model,
            "training_time": training_time,
            "model_name": model_name,
        }

        return model, training_time

    def evaluate_model(self, model, n_episodes=5):
        """Evaluate a trained model"""
        env = self.create_env()
        rewards = []

        for _ in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]

            rewards.append(episode_reward)

        env.close()
        return rewards

    def compare_policies(self):
        """Compare CNN and MLP policies"""
        print("\n" + "=" * 50)
        print("TRAINING COMPARISON RESULTS:")

        for policy_type, result in self.results.items():
            print(
                f"{policy_type} training time: {result['training_time']:.2f} seconds"
            )

        # Load best models for evaluation
        try:
            cnn_best = DQN.load("models/dqn_cnn/best_model")
            mlp_best = DQN.load("models/dqn_mlp/best_model")

            print("\nQuick evaluation (5 episodes each):")

            # Evaluate both models
            cnn_rewards = self.evaluate_model(cnn_best)
            mlp_rewards = self.evaluate_model(mlp_best)

            cnn_avg = sum(cnn_rewards) / len(cnn_rewards)
            mlp_avg = sum(mlp_rewards) / len(mlp_rewards)

            print(f"CNN Policy average reward: {cnn_avg:.2f}")
            print(f"MLP Policy average reward: {mlp_avg:.2f}")

            # Save the better model
            if cnn_avg > mlp_avg:
                print("\nCNN Policy performed better! Saving as dqn_model.zip")
                cnn_best.save("dqn_model")
                self.best_policy = "CnnPolicy"
            else:
                print("\nMLP Policy performed better! Saving as dqn_model.zip")
                mlp_best.save("dqn_model")
                self.best_policy = "MlpPolicy"

            return cnn_avg, mlp_avg

        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Saving CNN model as default")
            if "CnnPolicy" in self.results:
                self.results["CnnPolicy"]["model"].save("dqn_model")
                self.best_policy = "CnnPolicy"
            return None, None

    def train_all(self):
        """Train both CNN and MLP policies"""
        print(f"Starting DQN training on {self.env_name}")
        print(f"Total timesteps: {self.total_timesteps}")

        # Train CNN Policy
        print("\n" + "=" * 50)
        self.train_model("CnnPolicy", "dqn_cnn")

        # Train MLP Policy
        print("\n" + "=" * 50)
        self.train_model("MlpPolicy", "dqn_mlp")

        # Compare results
        self.compare_policies()

        print("\nTraining completed! Check 'models' and 'logs' directories.")
        print("Final model saved as 'dqn_model.zip'")

    def get_results(self):
        """Get training results"""
        return self.results


def main():
    """Main function"""
    # Create trainer instance
    trainer = DeepQ(
        env_name="ALE/Breakout-v5", total_timesteps=30000, seed=42
    )

    # Train both policies
    trainer.train_all()

    # Get results
    results = trainer.get_results()
    print(f"\nBest policy: {trainer.best_policy}")


if __name__ == "__main__":
    main()
