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

    def collect_trajectory(self, policy: Policy, state) -> list[tuple]:
        """collect trajectory from playing the game"""
        traj = []
        s, _, terminated = self.env.reset(state)
        while not terminated:
            a = policy(s)
            s_1, r_1, terminated = self.env.step(a)
            traj.append((s, a, r_1))
            s = s_1
        return traj

    def policy_evaluation(
            self, policy: callable, state, n_episode) -> dict[float]:
        """Collect trajectory starting from state and follow policy"""
        for episode in tqdm(range(n_episode)):
            traj = self.collect_trajectory(policy, state)
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


class MonteCarloOffPolicy:

    def __init__(self, env: Env, policy, weighted=True, seed=None):
        self.env = env
        self.V = self._make_state_value_function()
        self.C = self._make_state_value_function()
        self.count = self._make_state_value_function()
        self.pi = {s: policy(s) for s in self.V.keys()}  # target policy
        self.b = {"hit": 0.5, "stick": 0.5}  # behavior policy
        self.weighted = weighted
        self.rng = np.random.default_rng(seed)

    def _make_state_value_function(self) -> dict[float]:
        V = dict()
        for s in self.env.observation_space:
            V[s] = 0
        return V

    def collect_trajectory(self, policy: Policy, state) -> list[tuple]:
        """collect trajectory from playing the game"""
        traj = []
        s, _, terminated = self.env.reset(state)
        while not terminated:
            a = policy(s)
            s_1, r_1, terminated = self.env.step(a)
            traj.append((s, a, r_1))
            s = s_1
        return traj

    def policy_evaluation(self, state, n_episode) -> dict[float]:
        """Collect trajectory starting from state and follow policy"""
        def make_behavior_policy_from_dict():
            p = [p for a, p in self.b.items()]
            a = [a for a, p in self.b.items()]

            def policy(s):
                """Action probability is independent to state"""
                idx = self.rng.choice(len(p), p=p)
                return a[idx]

            return policy

        policy = make_behavior_policy_from_dict()
        v = []
        for episode in tqdm(range(n_episode), disable=True):
            traj = self.collect_trajectory(policy, state)
            self._value_update(traj)

            v.append(self.V[state])  # value history
        return v

    def _value_update(self, traj: Trajectory):
        """Update state, state-action value function and target policy
        at the same time
        Assume target policy is deterministic"""
        def weighted_update():
            # if W == 0, nothing happened
            self.C[s] += W
            self.V[s] += W / self.C[s] * (G - self.V[s])

        def ordinary_update():
            # if W == 0, still update V
            self.count[s] += 1
            self.V[s] += (W * G - self.V[s]) / self.count[s]

        G = 0
        W = 1
        while len(traj) > 0:
            s, a, r_1 = traj.pop()
            G = r_1 + 1 * G
            if self.weighted:
                if self.pi[s] == a:
                    W /= self.b[a]  # deterministic policy self.pi
                    weighted_update()
                else:
                    break
            else:  # simple average
                pi_p = 1 if self.pi[s] == a else 0
                W *= pi_p / self.b[a]  # deterministic policy self.pi
                ordinary_update()


def figure_5_3():

    def policy(s):
        return "hit" if s[0] < 20 else "stick"

    def state_policy_estimation(state):
        game = BlackJack(seed=123456)
        mc = MonteCarlo(game)
        v_ = 0
        while abs(v_ - v) > 1e-5:
            vd = mc.policy_evaluation(policy, state, n_episode=100000)
            v_ = vd[s]
            print(v_)

    def experiment(weighted=True):
        rng = np.random.default_rng(123456)
        V = np.empty((100, 10000))
        for run in tqdm(range(100)):
            game = BlackJack(seed=123456)
            mc = MonteCarloOffPolicy(
                game, policy, weighted=weighted, seed=rng.integers(100000))
            V[run, :] = mc.policy_evaluation(s, 10000)
        return V

    def mse(array):
        return np.power(array - v, 2).mean(axis=0)

    # Estimate the value of a state
    s = (13, True, 2)
    # v = state_policy_estimation(s)
    v = - 0.27726

    # Estimate by off (random) policy
    vw = experiment(weighted=True)
    vs = experiment(weighted=False)
    data = np.stack([mse(ar) for ar in (vw, vs)]).T

    # plot
    fig, ax = plt.subplots()
    x = np.arange(1, data.shape[0] + 1)
    ax.plot(x, data[:, 0], x, data[:, 1])
    ax.set_xscale('log')
    return data


if __name__ == '__main__':
    data = figure_5_3()
