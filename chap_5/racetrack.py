# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:52:54 2024

@author: ccw
"""
import numpy as np
from tqdm import tqdm

from track import Track, State, Action

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


type Policy = callable[State, Action]
type Trajectory = list[tuple[State, Action, int]]


class MonteCarlo:
    """weighted importance sampling.
    Behavior policy is uniform"""

    def __init__(self, track: Track, seed=None):
        self.env = track
        # value cannot be the same as reward
        self.Q = self._make_state_action_value_function(-100)
        self.C = self._make_state_action_value_function()
        self.pi = self._make_target_policy()
        self.rng = np.random.default_rng(seed)

    def _make_state_action_value_function(self, value=0):
        q = dict()
        for s, actions in self.env.action_space.items():
            q[s] = dict()
            for a in actions:
                q[s][a] = value
        return q

    def _make_target_policy(self):
        pi = dict()
        for s in self.Q.keys():
            pi[s] = self._greedy_action(s)
        return pi

    def _greedy_action(self, s):
        max_value = max(self.Q[s].values())
        for a, v in self.Q[s].items():
            if v == max_value:
                return a

    def collect_trajectory(self, policy: Policy) -> Trajectory:
        """collect trajectory from playing the game"""
        traj = []
        s, terminated = self.env.reset()
        while not terminated:
            a = policy(s)
            s_1, r_1, terminated = self.env.step(a)
            traj.append((s, a, r_1))
            s = s_1
        return traj

    def _uniform_policy(self, s) -> Action:
        """RETURN: action and its probability"""
        actions = self.env.action_space[s]
        a = tuple(self.rng.choice(actions))
        return a

    def policy_iteration(self, n_episode):
        """Collect trajectory starting from state and follow policy"""
        def behavior_policy(s):
            return self._uniform_policy(s)

        for episode in tqdm(range(n_episode), disable=False):
            traj = self.collect_trajectory(behavior_policy)
            self._value_update(traj)

    def _value_update(self, traj):
        """Update state, state-action value function and target policy
        at the same time
        Assume target policy is deterministic"""
        def weighted_update():
            self.C[s][a] += W
            self.Q[s][a] += W / self.C[s][a] * (G - self.Q[s][a])

        G = 0
        W = 1
        while len(traj) > 0:
            s, a, r_1 = traj.pop()
            G = r_1 + 1 * G
            weighted_update()
            self.pi[s] = self._greedy_action(s)
            if a != self.pi[s]:
                break
            pb = 1 / len(self.env.action_space[s])  # uniform policy
            W *= 1 / pb

    def trained_path(self, start=None, k=100) -> list[(int, int)]:
        """collect trajectory from playing the game"""
        count = 0
        traj = []
        s, terminated = self.env.reset(start)
        while not terminated and count < k:
            count += 1
            a = self.pi[s]
            s_1, r_1, terminated = self.env.step(a)
            traj.append((s, a, r_1))
            s = s_1

        path = [coor for (coor, _), _, _ in traj]
        path.append(s[0])  # finish coordinate
        return path


def plot_grid(track: np.ndarray):

    fig, ax = plt.subplots()

    # Setup grid world
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # background color
    rgb = np.ones(3)
    colors = [rgb * 0.2, rgb]
    custom_cmap = ListedColormap(colors)

    # plot
    _ = ax.pcolormesh(
        track, edgecolors="k", linewidth=0.5, cmap=custom_cmap)

    # adjust ticks
    # position
    ax.set_xticks(np.arange(track.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(track.shape[0]) + 0.5, minor=False)
    # hide the tick line
    ax.tick_params(axis='both', which='both', length=0)

    # set label
    ax.set_xticklabels(np.arange(track.shape[1]))
    ax.set_yticklabels(np.arange(track.shape[0]))

    return ax


def trace_plot(mc):

    def plot_path(ax, path, color):
        # adjust coordinate
        coors = np.array(path) + 0.5
        x = coors[:, 1]
        y = coors[:, 0]
        ax.plot(x, y, c=color)

    # plot
    ax = plot_grid(mc.env.track)
    # add trajectory
    for start, color in zip(mc.env._starting_line, ("blue", "red", "green")):
        path = mc.trained_path(start)
        plot_path(ax, path, color)


if __name__ == '__main__':
    track_ = np.array(
        [[0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 0]])

    track = Track(track_)
    mc = MonteCarlo(track)
    # print(mc.Q)
    # mc._greedy_action(((7, 4), (3, 1)))
    # mc.pi
    # mc._behavior_policy(((7, 4), (3, 1)))
    # traj = mc.collect_trajectory()
    mc.policy_iteration(10000)

    # plot trace
    trace_plot(mc)
