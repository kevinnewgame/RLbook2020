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
# from matplotlib import colors

type Env = BlackJack
type State = (int, bool, int)
type Action = str
type Policy = callable[State, Action]
type Trajectory = list[(State, Action, float)]


class MonteCarloES:

    def __init__(self, env: Env, policy: callable, seed=None):
        """policy: initial policy"""
        self.env = env
        self.Q = self._make_state_action_value_function()
        self.count = self._make_state_action_value_function()
        self.pi = {s: policy(s) for s in self.Q.keys()}
        self.rng = np.random.default_rng(seed)

    def _make_state_action_value_function(self) -> dict[dict[float]]:
        Q = dict()
        for s in self.env.observation_space:
            Q[s] = dict()
            for a in self.env.action_space:
                Q[s][a] = 0
        return Q

    def _uniform_sample(self, space: list):
        idx = self.rng.choice(len(space))
        return space[idx]

    def random_start(self) -> (State, Action):
        s = self._uniform_sample(self.env.observation_space)
        a = self._uniform_sample(self.env.action_space)
        return s, a

    def collect_trajectory(self, policy: Policy) -> list[tuple]:
        """collect trajectory from playing the game"""
        traj = []
        # exploring start
        s, a = self.random_start()
        _, _, _ = self.env.reset(s)
        s_1, r_1, terminated = self.env.step(a)
        traj.append((s, a, r_1))
        s = s_1
        while not terminated:
            a = policy(s)
            s_1, r_1, terminated = self.env.step(a)
            traj.append((s, a, r_1))
            s = s_1
        return traj

    def policy_iteration(self, n_episode):
        def policy(s):
            return self.pi[s]

        for episode in tqdm(range(n_episode)):
            traj = self.collect_trajectory(policy)
            self._value_update(traj)
        return deepcopy(self.pi), self._optimal_values()

    def _optimal_values(self):
        V = dict()
        for s in self.Q.keys():
            V[s] = self.Q[s][self.pi[s]]
        return V

    def _value_update(self, traj: Trajectory):
        """Update state, state-action value function and target policy
        at the same time
        Assume target policy is deterministic"""
        def greedy_action(s):
            max_value = max(self.Q[s].values())
            for a, v in self.Q[s].items():
                if v == max_value:
                    return a

        def update():
            self.count[s][a] += 1
            self.Q[s][a] += (G - self.Q[s][a]) / self.count[s][a]
            self.pi[s] = greedy_action(s)

        G = 0
        while len(traj) > 0:
            s, a, r_1 = traj.pop()
            G = r_1 + 1 * G
            if (s, a) not in [t_[0: 2] for t_ in traj]:  # first visit
                update()


def figure_5_2():
    """Don't know how many GPI to converge yet!"""

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

    def action2num(a: str) -> int:
        return 1 if a == "hit" else 0

    book_policy = np.array(
        [[1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # learning
    game = BlackJack(seed=123456)
    mc = MonteCarloES(game, policy, seed=654321)

    # policy evaluation
    count = 0
    policy_dict, _ = mc.policy_iteration(1)
    p_u, p_uu = to_array_value(policy_dict, action2num)
    # Need 2200000 iteration for the random seeds arbitrarily chosen!
    while abs(p_uu - book_policy).max() != 0:
        policy_dict, values = mc.policy_iteration(int(1e5))
        p_u, p_uu = to_array_value(policy_dict, action2num)
        count += int(1e5)
        print(count)
        print(p_uu)

        v_u, v_uu = to_array_value(values, lambda x: x)

    # plot
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    # _ = colors.Normalize(vmin=-1, vmax=1)

    axs[0, 0].imshow(p_u, origin='lower')
    axs[0, 1].imshow(v_u, origin='lower')
    axs[1, 0].imshow(p_uu, origin='lower')
    axs[1, 1].imshow(v_uu, origin='lower')
    # fig.colorbar(im, ax=axs)

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
    return mc


if __name__ == '__main__':
    mc = figure_5_2()
