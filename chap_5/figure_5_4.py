# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:25:45 2024

@author: ccw

https://ai.stackexchange.com/questions/22372/should-the-importance-sampling-ratio-be-updated-at-the-end-of-the-for-loop-in-th
I had the same question too, but finally, I found it was just a mismatch between the explained context and the algorithm given.
Note that in context, it explains importance sampling ratio for state value function (V) instead of state-action value function (Q) which is the target of the boxed algorithm. Following exercise 5.6, you will clear our misunderstanding.
"""
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import matplotlib.pyplot as plt
# from matplotlib import colors


class Loop:

    def __init__(self, seed=None):
        self.observation_space = np.arange(2)
        self.action_space = ["left", "right"]
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = 0
        self.terminated = False
        return (self.state, self.terminated)

    def step(self, a):
        """RETURN: state, reward, terminated"""
        if self.terminated:
            raise EOFError("Reset the game before stepping.")
        if a not in self.action_space:
            raise KeyError("Give valid action!")

        if a == "left":
            if self.rng.random() <= 0.1:
                self.state = 1
                self.terminated = True
                return (self.state, 1, self.terminated)
            else:
                return (self.state, 0, self.terminated)
        elif a == "right":
            self.state = 1
            self.terminated = True
            return (self.state, 0, self.terminated)


class MonteCarloOffPolicy:

    def __init__(self, env, weighted=True, seed=None):
        self.env = env
        self.V = self._make_state_value_function()
        self.count = self._make_state_value_function()
        self.C = self._make_state_value_function()
        self.pi = {0: "left"}
        self.b = {"left": 0.5, "right": 0.5}  # independent to state
        self.weighted = weighted
        self.rng = np.random.default_rng(seed)

    def _make_state_value_function(self) -> dict[float]:
        V = dict()
        for s in self.env.observation_space:
            V[s] = 0
        return V

    def collect_trajectory(self, policy) -> list[tuple]:
        """collect trajectory from playing the game"""
        traj = []
        s, terminated = self.env.reset()
        while not terminated:
            a = policy(s)
            s_1, r_1, terminated = self.env.step(a)
            traj.append((s, a, r_1))
            s = s_1
        return traj

    def policy_evaluation(self, n_episode) -> dict[float]:
        """Collect trajectory starting from state and follow policy"""
        policy = self._make_behavior_policy_from_dict()
        v = []
        for episode in tqdm(range(n_episode), disable=True):
            traj = self.collect_trajectory(policy)
            self._value_update(traj)

            v.append(self.V[0])  # value history
        return v

    def _make_behavior_policy_from_dict(self):
        p = [p for a, p in self.b.items()]
        a = [a for a, p in self.b.items()]

        def policy(s):
            """Action probability is independent to state"""
            idx = self.rng.choice(len(p), p=p)
            return a[idx]

        return policy

    def _value_update(self, traj):
        """Update state, state-action value function and target policy
        at the same time
        Assume target policy is deterministic"""
        def weighted_update():
            # if W == 0, nothing happened
            self.C[s] += W
            self.V[s] += W / self.C[s] * (G - self.V[s])

        def simple_update():
            # if W == 0, still update V
            self.count[s] += 1
            self.V[s] += (W * G - self.V[s]) / self.count[s]

        G = 0
        W = 1
        while len(traj) > 0:
            s, a, r_1 = traj.pop()
            G = r_1 + 1 * G
            pi_p = 1 if self.pi[s] == a else 0
            W *= pi_p / self.b[a]  # deterministic policy self.pi
            if self.weighted:
                if W == 0:
                    break
                else:
                    if s not in [t_[0] for t_ in traj]:  # first visit
                        weighted_update()
            else:  # simple average
                if s not in [t_[0] for t_ in traj]:  # first visit
                    simple_update()


def figure_5_4(n_episode):

    def experiment(weighted=True, seed=None):
        game = Loop(seed=123456)
        mc = MonteCarloOffPolicy(game, weighted=weighted, seed=seed)
        return mc.policy_evaluation(n_episode)

    fig, ax = plt.subplots()
    rng = np.random.default_rng(123456)
    for run in tqdm(range(10)):
        # Estimate by off (random) policy
        vs = experiment(weighted=False, seed=rng.integers(100000))
        # vw = experiment(weighted=True, seed=rng.integers(100000))

        data = vs
        x = np.arange(len(data)) + 1
        ax.plot(x, data)

    ax.set_xscale('log')
    ax.set_ylim(0, 3)


if __name__ == '__main__':
    data = figure_5_4(1000000)
