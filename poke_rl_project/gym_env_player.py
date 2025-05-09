# gym_env_player.py
from poke_env.player import Player
from poke_env.environment.abstract_battle import AbstractBattle
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import numpy as np
from simple_rl_player import SimpleRLPlayer


class GymEnvPlayer(Env):
    def __init__(self, opponent: Player):
        self.player = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent)
        self.action_space = Discrete(4)
        self.observation_space = Box(low=-1, high=4, shape=(10,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.player.reset_battles()
        while self.player._current_battle is None:
            self.player._async_play()
        obs = self.player.embed_battle(self.player._current_battle)
        return obs, {}

    def step(self, action):
        battle = self.player._current_battle
        move = battle.available_moves[action] if action < len(battle.available_moves) else battle.available_moves[0]
        self.player._send_move(move)
        self.player._async_play()

        obs = self.player.embed_battle(self.player._current_battle)
        reward = self.player.calc_reward(battle)
        done = battle.finished
        return obs, reward, done, False, {}