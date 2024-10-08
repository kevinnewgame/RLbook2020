# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:09:14 2024

@author: ccw

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


MOVE_UP = (-1, 0)
MOVE_RIGHT = [0, 1]
MOVE_DOWN = [1, 0]
MOVE_LEFT = [0, -1]

ACTION2MOVE = {"up": MOVE_UP,
               "down": MOVE_DOWN,
               "right": MOVE_RIGHT,
               "left": MOVE_LEFT}


class GridWorldModel:

    def __init__(self):
        self.shape = (4, 4)
        self.goal = [(0, 0), (3, 3)]
        self.observation_space = np.arange(0, 16)  # includes terminal states
        self.S = np.arange(1, 15)  # non-terminal states
        self.action_space = ["up", "down", "right", "left"]

    def step(self, s: int, a: str) -> list[float, int, int]:
        """Given current state and action, return next probabiltiy, state and
        reward"""
        # Input checking
        if s not in self.S:
            raise ValueError("Wrong state!")
        if a not in self.action_space:
            raise ValueError("Wrong action!")

        coor = self.state2coor(s)
        # move
        row_1, col_1 = np.array(coor) + np.array(ACTION2MOVE[a])
        coor_1 = (row_1, col_1)
        try:
            s_1 = self.coor2state(coor_1)
        except ValueError:  # outside of the grid
            s_1 = s
        return [(1, s_1, -1)]

    def state2coor(self, s: int) -> (int, int):
        return np.unravel_index(s, self.shape)

    def coor2state(self, coor: (int, int)) -> int:
        return np.ravel_multi_index(coor, self.shape)


class DP:

    def __init__(self, model: GridWorldModel, discount_rate: float = 1):
        self.model = model
        self.gamma = discount_rate
        self.V = self._make_value_function()

    def _make_value_function(self):
        V = dict()
        for s in self.model.observation_space:
            V[s] = 0
        return V

    def state_action_value(self, s, a):
        q = 0
        for p, s_1, r_1 in self.model.step(s, a):
            q += p * (r_1 + self.gamma * self.V[s_1])
        return q

    def state_value_update(self, policy: dict):
        """value replacement in one array thus different process from the book
        """
        for s in self.model.S:
            v = 0
            for a in self.model.action_space:
                v += policy[s][a] * self.state_action_value(s, a)
            self.V[s] = v

    def policy_evaluation(self, policy: dict, eps=1e-7):
        while True:
            V_old = self.V.copy()
            self.state_value_update(policy)
            delta = 0
            for s, v in self.V.items():
                delta = max(delta, abs(V_old[s] - v))
            if delta < eps:
                break

    def show_value_function(self):
        V = np.empty(self.model.shape)
        for s, v in self.V.items():
            r, c = self.model.state2coor(s)
            V[r, c] = v
        print(V)

    def one_step_search(self, s: int) -> dict[str: float]:
        return {a: self.state_action_value(s, a)
                for a in self.model.action_space}

    def find_optimal_actions(self, s: int, decimal=4) -> list[str]:
        action_value = self.one_step_search(s)
        max_value = max(action_value.values())
        return [a for a, v in action_value.items()
                if round(v, 4) == round(max_value, 4)]

    def plot_greedy_policy(self):

        def action2dydx(a, velocity=0.25):
            dy, dx = [m * velocity for m in ACTION2MOVE[a]]
            return dy, dx

        def plot_arrow(ax, x, y, dx, dy):
            ax.arrow(x, y, dx, dy,
                     head_width=0.1, head_length=0.1,
                     fc='r', ec='r')

        # fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        fig, ax = plt.subplots()

        # grid plot
        # Setup grid world
        ax.set_aspect('equal')
        ax.invert_yaxis()
        # background color: white
        colors = [(1, 1, 1)]  # only white
        custom_cmap = ListedColormap(colors)

        data = np.zeros(self.model.shape)
        _ = ax.pcolormesh(
            data, edgecolors='k', linewidth=0.5, cmap=custom_cmap)

        # Hide the tick and label
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

        # arrows
        for s in self.model.S:
            y, x = self.model.state2coor(s)
            actions = self.find_optimal_actions(s)
            for a in actions:
                dy, dx = action2dydx(a)
                plot_arrow(ax, x + 0.5, y + 0.5, dx, dy)


if __name__ == '__main__':
    np.set_printoptions(precision=1)

    model = GridWorldModel()
    print("Observation space:", model.observation_space)
    print("Action space:", model.action_space)
    assert 1 == model.coor2state(model.state2coor(1))
    for prob, r, s in model.step(s=1, a="up"):
        print(f"Move up from state 1 then get {r} reward, to the state {s}",
              f"with probability {prob}")

    # make policy
    policy = dict()
    for s in model.observation_space:
        policy[s] = dict()
        for a in model.action_space:
            policy[s][a] = 1 / len(model.action_space)

    # policy evaluation
    dp = DP(model)
    for _ in range(10):
        dp.state_value_update(policy)
        dp.show_value_function()

    dp.policy_evaluation(policy)
    dp.show_value_function()
    dp.plot_greedy_policy()
