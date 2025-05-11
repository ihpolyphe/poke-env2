from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from poke_env.player.random_player import RandomPlayer
from gym_env_player import GymEnvPlayer
from callbacks import FullTrainingLogger

def main():
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = GymEnvPlayer()
    train_env.player.opponent = opponent  # opponentを後から設定

    eval_opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = GymEnvPlayer()
    eval_env.player.opponent = eval_opponent

    win_logger = FullTrainingLogger(eval_env, eval_freq=1000, n_eval_episodes=5)

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=1
    )
    model.learn(total_timesteps=20000, callback=win_logger)
    model.save("dqn_pokemon_model")

    # 勝率のプロット表示
    win_logger.plot_all()

if __name__ == "__main__":
    main()