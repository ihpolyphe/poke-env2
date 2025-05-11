# gym_env_player.py
import asyncio
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from poke_env.player import Player
from poke_env.environment.abstract_battle import AbstractBattle
from simple_rl_player import SimpleRLPlayer # reward, embed_battle を定義したクラスを想定

class GymEnvPlayer(Env):
    def __init__(self):
        super().__init__()
        self.player = SimpleRLPlayer(
            battle_format="gen8randombattle",
        )
        self.action_space = Discrete(4)
        self.observation_space = Box(low=-1, high=4, shape=(10,), dtype=np.float32)
        self._current_battle = None

    def reset(self, seed=None, options=None):
        self.player.reset_battles()

        # バトルが開始するまで待機
        while len(self.player.battles) == 0:
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))

        # 最初のバトルを取得
        self._current_battle = next(iter(self.player.battles.values()))
        obs = self.player.embed_battle(self._current_battle)
        return obs, {}

    def step(self, action: int):
        battle = self._current_battle

        if action < len(battle.available_moves):
            move = battle.available_moves[action]
        else:
            move = battle.available_moves[0]

        self.player._send_move(move)

        while not battle.finished and not battle.can_execute_command:
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))

        obs = self.player.embed_battle(battle)
        reward = self.player.calc_reward(battle)
        done = battle.finished
        return obs, reward, done, False, {}