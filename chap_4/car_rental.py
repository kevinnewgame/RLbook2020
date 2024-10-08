# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:05:26 2024

@author: ccw
"""
import numpy as np
from scipy.stats import poisson


class CarRentalModel:

    def __init__(self, different_cost=False):
        self.n_parking = 20
        self.observation_space = self._make_obs()
        self.action_space = np.arange(-5, 5 + 1)

        (self.r_1, self.t_1), (self.r_2, self.t_2) = \
            self.make_transition_matrix()

        self.different_cost = different_cost

    def _make_obs(self):
        states = []
        for n_1 in range(self.n_parking + 1):
            for n_2 in range(self.n_parking + 1):
                states.append((n_1, n_2))
        return states

    def step(self, s: (int, int), a: int) -> (float, np.ndarray):
        """For DP, q(s, a) can be seen as addtion of two parts
        q(s, a) = r(s, a) + sum by s': p(s'|s, a) * V[s']
        Thus, we provide r(s, a) and p(s'|s, a) as return for
        state value update"""
        def extra_parking_cost(n):
            if n > 10:
                return 4
            return 0

        # 1. moving car
        moving_cost = 2
        n_1, n_2 = self.move_car(s, a)
        r = - moving_cost * abs(a)
        if self.different_cost:
            r = - moving_cost * max(abs(a) - 1, 0)  # one is free
            r += - sum(extra_parking_cost(n) for n in (n_1, n_2))

        # 2. car request
        rent_earn = 10
        r += (self.r_1[n_1] + self.r_2[n_2]) * rent_earn
        # 3. final transition after car return
        t = self.t_1[n_1, :].reshape(-1, 1) @ self.t_2[n_2, :].reshape(1, -1)
        return r, t

    def move_car(self, s: (int, int), a: int) -> (int, int):
        """move n_move cars from L1 (with n_1 cars) to L2 (n_2)
        Should consider available cars in from place and
        the empty parking lot in to place.
        RETURN: # cars in L1, # cars in L2, # cars actually move
        """
        empty_0, empty_1 = [self.n_parking - n for n in s]
        if a > 0:  # move car from L1 to L2
            can_move = min(s[0], a, empty_1)
        elif a < 0:  # from L2 to L1
            can_move = - min(s[1], abs(a), empty_0)
        else:
            return s  # do nothing when a is zero
        return s[0] - can_move, s[1] + can_move

    def after_request(self, n: int, mu: float) -> (float, int, int):
        """Given number of car in one place, return the
        1. the probability of events(different requests)
        2. number of car rent to(<= actual request)
        3. number of car in a place
        after customer request"""
        rv = poisson(mu)
        rent = np.arange(n + 1)
        p = rv.pmf(rent)
        n_1 = n - rent
        p[-1] = 1 - p[:-1].sum()
        return p, rent, n_1

    def after_return(self, n: int, mu: float) -> (float, int):
        """Given number of car in one place, return the
        1. the probability of events(different returns)
        3. number of car in a place
        after customer return"""
        rv = poisson(mu)
        return_ = np.arange(0, self.n_parking - n + 1)
        p = rv.pmf(return_)
        n_1 = return_ + n
        p[-1] = 1 - p[:-1].sum()
        return p, n_1

    def make_transition_matrix(self) -> \
            ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        """Pre-calculate the expected reward (car rent) and transition matrix
        for stepping for each location"""
        def request_transition(mu) -> (float, np.ndarray):
            """RETURN: expected reward and transition matrix(probability)
            for each starting n car."""
            def make_transition_vector_from_p(p: np.ndarray) -> np.ndarray:
                return np.pad(np.flip(p),
                              (0, self.n_parking + 1 - len(p)),
                              constant_values=0)

            t = np.empty((self.n_parking + 1, self.n_parking + 1))
            r = np.empty(self.n_parking + 1)
            for n in range(self.n_parking + 1):
                p, rent, _ = self.after_request(n, mu)
                r[n] = (p * rent).sum()
                t[n, :] = make_transition_vector_from_p(p)
            return r, t

        def return_transition(mu) -> np.ndarray:
            def make_transition_vector_from_p(p: np.ndarray) -> np.ndarray:
                return np.pad(p,
                              (self.n_parking + 1 - len(p), 0),
                              constant_values=0)

            t = np.empty((self.n_parking + 1, self.n_parking + 1))
            for n in range(t.shape[0]):
                p, _ = self.after_return(n, mu)
                t[n, :] = make_transition_vector_from_p(p)
            return t

        def expected_reward_and_transition(mu_req, mu_ret):
            r, t_req = request_transition(mu_req)  # 1. car request
            t_ret = return_transition(mu_ret)  # 2. car return
            return r, t_req @ t_ret

        # After moving car at night, the transition of location 1 and 2 are
        # independent, thus we can calculate them seperately.
        r_1, t_1 = expected_reward_and_transition(3, 3)
        r_2, t_2 = expected_reward_and_transition(4, 2)
        return (r_1, t_1), (r_2, t_2)


class DP:

    def __init__(self, model: CarRentalModel):
        self.model = model
        n = self.model.n_parking + 1
        self.V = np.zeros((n, n))
        # give initial policy as zero
        self.pi = np.zeros(self.V.shape, dtype=int)

    def q_value(self, s: (int, int), a: int):
        r, t = self.model.step(s, a)
        return r + 0.9 * (t * self.V).sum()

    def value_update(self):
        for s in self.model.observation_space:
            a = self.pi[s[0], s[1]]
            self.V[s[0], s[1]] = self.q_value(s, a)

    def policy_evaluation(self, eps=1e-2):
        while True:
            V_old = self.V.copy()
            self.value_update()
            if abs(V_old - self.V).max() < eps:
                break

    def policy_update(self) -> bool:
        """Find better policy for each state given the value function"""
        def find_greedy_action(s):
            vs = [self.q_value(s, a) for a in self.model.action_space]
            return np.array(vs).argmax() - 5  # starts from -5

        policy_change = False
        for s in self.model.observation_space:
            a = find_greedy_action(s)
            if self.pi[s[0], s[1]] != a:
                policy_change = True
                self.pi[s[0], s[1]] = a
        return policy_change

    def policy_improvement(self) -> (np.ndarray, np.ndarray):
        """update policy function (of each state)"""
        self.policy_evaluation()
        while True:
            policy_change = self.policy_update()
            if policy_change:
                self.policy_evaluation()
            else:
                return self.V.copy(), self.pi.copy()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def heatmaps(v, pi):
        def heatmap(ax, data):
            im = ax.imshow(data, origin='lower')
            ax.figure.colorbar(im, ax=ax)

        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        heatmap(axs[0], pi)
        heatmap(axs[1], v)

    model = CarRentalModel()
    assert (0, 12) == model.move_car((5, 7), 6)
    model.after_request(5, 3)
    model.after_return(12, 4)

    model.make_transition_matrix()
    model.step((10, 10), 3)

    agent = DP(model)
    value, policy = agent.policy_improvement()
    heatmaps(value, policy)

    # Exercise
    model = CarRentalModel(different_cost=True)
    agent = DP(model)
    value, policy = agent.policy_improvement()
    heatmaps(value, policy)
