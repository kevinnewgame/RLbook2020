# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:03:55 2024

@author: ccw
"""
import numpy as np
from itertools import product
from typing import Tuple


State = Tuple[int, bool, int]


class BlackJack:

    possible_cards = np.arange(1, 14)

    class Player:

        def __init__(self):
            self.point = 0
            self.usable_ace = False
            self.bust = False

        def receive(self, card: int):
            if self.bust:
                raise EOFError(
                    "The player is bust and cannot receive more card.")
            if card not in BlackJack.possible_cards:
                raise ValueError(
                    "Invalid card.")

            # card to point
            if card > 10:  # J, Q, and K
                point = 10
            elif card == 1:
                if self.point + 11 <= 21:
                    point = 11
                    self.usable_ace = True
                else:
                    point = 1
            else:
                point = card
            # add point
            p_1 = self.point + point
            if p_1 > 21:
                if self.usable_ace:
                    p_1 -= 10
                    self.usable_ace = False
                else:
                    self.bust = True
            self.point = p_1

        def __repr__(self):
            return f"Points: {self.point}  Usable Ace: {self.usable_ace}  " \
                f"Bust: {self.bust}"

    def __init__(self, seed=None):
        """Two cards for each player. One card of dealer's is flipped up"""
        # player's point, usable ace, dealer's flipped up
        self.observation_space = tuple(
            product(range(12, 22), [True, False], range(1, 11)))
        self.action_space = ["hit", "stick"]

        self.rng = np.random.default_rng(seed=seed)

    def deal_card(self) -> int:
        return self.rng.choice(__class__.possible_cards)

    def reset(self, s=None):
        self.player = BlackJack.Player()
        self.dealer = BlackJack.Player()
        if isinstance(s, tuple):  # specific start
            # player side
            point, usable_ace, self.flip_up = s
            self.player.point = point
            self.player.usable_ace = usable_ace
            # dealer side
            self.dealer.receive(self.flip_up)
        elif s is None:  # random start
            # player side
            # before 11, we don't need to consider stick
            while self.player.point <= 11:
                self.player.receive(self.deal_card())
            # dealer side
            card = self.deal_card()
            self.flip_up = card if card <= 10 else 10
            self.dealer.receive(self.flip_up)
        self.terminated = False
        return self._return_observation(0)
        # natural (We don't care about the natural case since there no action
        # to take)
        # if self.player.point == 21:
        #     if self.dealer.point < self.player.point:
        #         self.terminated = True
        #         return self._return_observation(1)
        #     elif self.dealer.point == self.player.point:
        #         self.terminated = True
        #         return self._return_observation(0)
        # return self._return_observation(0)

    def step(self, a):
        if self.terminated:
            raise EOFError("Reset the game before stepping.")
        if a not in self.action_space:
            raise KeyError("Give valid action!")

        if a == "hit":  # player's turn
            self.player.receive(self.deal_card())
            if self.player.bust:
                self.terminated = True
                return self._return_observation(-1)
            return self._return_observation(0)

        elif a == "stick":  # dealer's turn
            while self.dealer.point < 17:
                self.dealer.receive(self.deal_card())
            # compare the final result
            self.terminated = True
            if self.dealer.bust:
                return self._return_observation(1)
            if self.player.point > self.dealer.point:
                return self._return_observation(1)
            elif self.player.point == self.dealer.point:
                return self._return_observation(0)
            elif self.player.point < self.dealer.point:
                return self._return_observation(-1)

    def _return_observation(self, reward) -> (State, int, bool):
        """return the state and reward"""
        return ((self.player.point, self.player.usable_ace, self.flip_up),
                reward,
                self.terminated)


if __name__ == '__main__':
    player = BlackJack.Player()
    print(player)
    player.receive(13)
    print(player)
    player.receive(1)
    print(player)
    player.receive(2)
    print(player)
    player.receive(10)
    print(player)
    # player.receive(3)
    print(player)

    # game = BlackJack(seed=0)
    # count = np.empty(14)
    # for _ in range(int(1e5)):
    #     count[game.deal_card()] += 1

    game = BlackJack(seed=0)
    game.observation_space
    (obs, r, terminated) = game.reset()
    (obs, r, terminated) = game.step("hit")
    (obs, r, terminated) = game.step("stick")
    game.dealer.point
