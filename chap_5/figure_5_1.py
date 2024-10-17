# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:25:45 2024

@author: ccw
"""
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from blackJack import BlackJack

import matplotlib.pyplot as plt
from matplotlib import colors

type Env = BlackJack
type State = (int, bool, int)
type Action = str
type Policy = callable[State, Action]
type Trajectory = list[(State, Action, float)]


class MonteCarlo:

    def __init__(self, env: Env):
        self.env = env
        self.V = self._make_state_value_function()
        self.count = self._make_state_value_function()

    def _make_state_value_function(self) -> dict[float]:
        V = dict()
        for s in self.env.observation_space:
            V[s] = 0
        return V

    def collect_trajectory(self, policy: Policy) -> list[tuple]:
        """collect trajectory from playing the game"""
        traj = []
        s, _, terminated = self.env.reset()
        while not terminated:
            a = policy(s)
            s_1, r_1, terminated = self.env.step(a)
            traj.append((s, a, r_1))
            s = s_1
        return traj

    def policy_evaluation(self, policy: callable, n_episode) -> dict[float]:
        for episode in tqdm(range(n_episode)):
            traj = self.collect_trajectory(policy)
            self._value_update(traj)
        return deepcopy(self.V)

    def _value_update(self, traj: Trajectory):
        """Update state, state-action value function and target policy
        at the same time
        Assume target policy is deterministic"""
        def update_v():
            self.count[s] += 1
            self.V[s] += (G - self.V[s]) / self.count[s]

        G = 0
        while len(traj) > 0:
            s, _, r_1 = traj.pop()
            G = r_1 + 1 * G
            if s not in [t_[0] for t_ in traj]:  # first visit
                update_v()


def figure_5_1():

    def dict2array(d: dict, func: callable) -> np.ndarray:
        array = np.empty((10, 10))
        for (point, _, flip), v in d.items():
            array[point - 12, flip - 1] = func(v)
        return array

    def to_array_value(v: dict, func: callable) -> tuple[np.ndarray]:
        # split into usable_ace and non-usable_ace
        t = {s: value for s, value in v.items() if s[1]}
        f = {s: value for s, value in v.items() if not s[1]}
        # transform to array
        u, uu = [dict2array(d, func) for d in (t, f)]  # usable, unusable
        return u, uu

    def policy(s):
        return "hit" if s[0] < 20 else "stick"

    # learning
    game = BlackJack(seed=123456)
    mc = MonteCarlo(game)
    # policy evaluation
    v = []
    v.append(mc.policy_evaluation(policy, 10000))
    v.append(mc.policy_evaluation(policy, 490000))

    # plot
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    # !!!
    _ = colors.Normalize(vmin=-1, vmax=1)

    u_0, uu_0 = to_array_value(v[0], lambda x: x)
    im = axs[0, 0].imshow(u_0, origin='lower')
    axs[1, 0].imshow(uu_0, origin='lower')

    u_1, uu_1 = to_array_value(v[1], lambda x: x)
    axs[0, 1].imshow(u_1, origin='lower')
    axs[1, 1].imshow(uu_1, origin='lower')
    fig.colorbar(im, ax=axs)

    # set ticks
    for row in axs:
        for ax in row:
            ax.set_xticks(np.arange(1, 11) - 1)
            labels = [str(i) for i in np.arange(1, 11)]
            labels[0] = "A"
            ax.set_xticklabels(labels)

            ax.set_yticks(np.arange(12, 22) - 12)
            labels = [str(i) for i in np.arange(12, 22)]
            ax.set_yticklabels(labels)

    # fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # u_0, uu_0 = to_array_value(v[0])

    # from mpl_toolkits.mplot3d import axes3d
    # X, Y, Z = axes3d.get_test_data(0.05)

    # axs[0, 0].plot_wireframe(np.arange(1, 11), np.arange(12, 22), u_0)
    # im = axs[0, 0].imshow(u_0, origin='lower')
    # axs[1, 0].imshow(uu_0, origin='lower')

    # u_1, uu_1 = to_array_value(v[1])
    # axs[0, 1].imshow(u_1, origin='lower')
    # axs[1, 1].imshow(uu_1, origin='lower')
    # fig.colorbar(im, ax=axs)
    return v


if __name__ == '__main__':
    v = figure_5_1()
