import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from callbacks import FullTrainingLogger
from poke_env.environment.abstract_battle import AbstractBattle
from gymnasium.spaces import Space, Box
from poke_env.player import (
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)
from poke_env.data import GenData
from poke_env.player import RandomPlayer
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from showdown_format import TEAM
import sys

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, team: str = "", **kwargs):
        if team is None:
            # self._team = RandomTeambuilder()  # ランダムチーム
            # ランダム方法は定義できないので一旦exit
            print("!!! チームをインポートしてください !!!")
            sys.exit()
        else:
            self._team = ConstantTeambuilder(team)  # 固定チーム
            print("import set team")
            print(self._team)
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        self.observation_spaces = Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
        )

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(8).type_chart,   
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def current_battle(self):
        if self._battles:
            return list(self._battles.values())[-1]  # 最新のバトルを取得
        return None

# そのままrandomクラスを使用するとteam属性が定義されておらずエラーになるので
# 自作したクラスを使用する
class MyRandomPlayer(RandomPlayer):
    def __init__(self, battle_format: str = "gen8randombattle", team: str = ""):
        self._team = ConstantTeambuilder(team)
        super().__init__(battle_format=battle_format)

def train():
    opponent = MyRandomPlayer(battle_format="gen8randombattle",team=TEAM)
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True,
        team=TEAM
    )

    opponent = MyRandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True,
        team=TEAM
    )
    opponent = MyRandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True,
        team=TEAM
    )

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
    train()