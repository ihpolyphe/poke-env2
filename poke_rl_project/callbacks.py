import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback

class FullTrainingLogger(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        # ログ用変数
        self.steps = []
        self.win_rates = []
        self.mean_rewards = []
        self.mean_lengths = []
        self.action_distributions = defaultdict(list)

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            wins = 0
            rewards = []
            lengths = []
            action_counts = np.zeros(self.model.action_space.n)

            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0
                ep_length = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    action_counts[action] += 1
                    obs, reward, done, _, _ = self.eval_env.step(action)
                    ep_reward += reward
                    ep_length += 1

                rewards.append(ep_reward)
                lengths.append(ep_length)
                if ep_reward > 0:
                    wins += 1

            # ログに記録
            win_rate = wins / self.n_eval_episodes
            mean_reward = np.mean(rewards)
            mean_length = np.mean(lengths)

            self.steps.append(self.n_calls)
            self.win_rates.append(win_rate)
            self.mean_rewards.append(mean_reward)
            self.mean_lengths.append(mean_length)
            for a in range(self.model.action_space.n):
                self.action_distributions[a].append(action_counts[a] / np.sum(action_counts))

            if self.verbose:
                print(f"[Step {self.n_calls}] WinRate: {win_rate:.2f} | Reward: {mean_reward:.2f} | Steps: {mean_length:.1f}")

        return True

    def plot_all(self):
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        plt.plot(self.steps, self.win_rates, label="Win Rate", color="green")
        plt.ylabel("Win Rate")
        plt.xlabel("Steps")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.steps, self.mean_rewards, label="Mean Reward", color="blue")
        plt.ylabel("Mean Reward")
        plt.xlabel("Steps")
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(self.steps, self.mean_lengths, label="Mean Episode Length", color="orange")
        plt.ylabel("Steps per Episode")
        plt.xlabel("Steps")
        plt.grid(True)

        plt.subplot(2, 2, 4)
        for a, values in self.action_distributions.items():
            plt.plot(self.steps, values, label=f"Action {a}")
        plt.xlabel("Steps")
        plt.ylabel("Action Frequency")
        plt.legend()
        plt.grid(True)

        plt.suptitle("Training Metrics")
        plt.tight_layout()
        plt.savefig("training_metrics.png")
        plt.show()