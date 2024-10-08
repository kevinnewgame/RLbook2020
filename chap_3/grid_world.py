# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:37:02 2024

@author: ccw
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


MOVE_UP = [-1, 0]
MOVE_RIGHT = [0, 1]
MOVE_DOWN = [1, 0]
MOVE_LEFT = [0, -1]

ACTION2MOVE = {"up": MOVE_UP,
               "down": MOVE_DOWN,
               "right": MOVE_RIGHT,
               "left": MOVE_LEFT}


class GridWorldModel:

    def __init__(self):
        self.shape = (5, 5)
        self.observation_space = np.arange(25)
        self.S = self.observation_space.copy()
        self.action_space = ["up", "down", "right", "left"]

    def step(self, s: int, a: str) -> list[float, int, int]:
        """Given current state and action, return next probabiltiy, state and
        reward"""
        # Input checking
        if s not in self.observation_space:
            raise ValueError("Wrong state!")
        if a not in self.action_space:
            raise ValueError("Wrong action!")

        coor = self.state2coor(s)
        if coor == (0, 1):  # A
            s_1 = self.coor2state((4, 1))
            r_1 = 10
        elif coor == (0, 3):  # B
            s_1 = self.coor2state((2, 3))
            r_1 = 5
        else:  # regular move
            r_1, c_1 = np.array(coor) + np.array(ACTION2MOVE[a])
            coor_1 = (r_1, c_1)
            try:
                s_1 = self.coor2state(coor_1)
                r_1 = 0
            except ValueError:  # outside of the grid
                s_1 = s
                r_1 = -1
        return [(1, s_1, r_1)]

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

    def optimal_state_value_update(self):
        """value replacement in one array thus different process from the book
        """
        for s in self.model.S:
            max_v = max([self.state_action_value(s, a)
                         for a in self.model.action_space])
            self.V[s] = max_v

    def policy_evaluation(self, policy: dict, eps=1e-7):
        while True:
            V_old = self.V.copy()
            self.state_value_update(policy)
            delta = 0
            for s, v in self.V.items():
                delta = max(delta, abs(V_old[s] - v))
            if delta < eps:
                break

    def optimal_state_value(self, eps=1e-7):
        while True:
            V_old = self.V.copy()
            self.optimal_state_value_update()
            delta = 0
            for s, v in self.V.items():
                delta = max(delta, abs(V_old[s] - v))
            if delta < eps:
                break

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

    def show_value_function(self):
        V = np.empty(self.model.shape)
        for s, v in self.V.items():
            r, c = self.model.state2coor(s)
            V[r, c] = v
        print(V)


if __name__ == '__main__':
    np.set_printoptions(precision=1)

    model = GridWorldModel()
    print("Observation space:", model.observation_space)
    print("Action space:", model.action_space)

    assert 1 == model.coor2state(model.state2coor(1))

    # test pairs
    sapairs = [(0, "up"), (1, "down"), (3, "left"), (2, "right")]
    for s, a in sapairs:
        for prob, s_1, r_1 in model.step(s, a):
            print(f"Move {a} from state {s} then get {r_1} reward,",
                  f"to the state {s_1} with probability {prob}")

    # make policy (equal likely policy)
    policy = dict()
    for s in model.S:
        policy[s] = dict()
        for a in model.action_space:
            policy[s][a] = 1 / len(model.action_space)

    # policy evaluation
    dp = DP(model, discount_rate=0.9)
    dp.policy_evaluation(policy)
    dp.show_value_function()

    # action_value = dp.one_step_search(1)
    # action_value = dp.one_step_search(5)

    # plot deterministic policy
    dp.plot_greedy_policy()

    # optimal state value and policy
    dp.optimal_state_value()
    dp.show_value_function()
    dp.plot_greedy_policy()
