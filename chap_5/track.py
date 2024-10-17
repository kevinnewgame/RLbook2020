# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:05:39 2024

@author: ccw
"""
import numpy as np
from itertools import product
from io import StringIO

type Action = (int, int)
type State = ((int, int), (int, int))


class Track:

    def __init__(self, track: np.ndarray, seed=None):
        self.track = track
        self.shape = track.shape
        self._starting_line = self._make_start_line()
        self._finish_line = self._make_finish_line()

        self.observation_space, self.action_space = self._make_spaces()

        self._rng = np.random.default_rng(seed)

    def reset(self, coor=None) -> ((State, bool), list[Action]):
        """(row direction, column direction)
        RETURN: state, terminated, and possible actions"""
        # starting coordinate
        starts = self._starting_line
        self.coor = starts[self._rng.choice(len(starts))]
        if coor is not None:
            self.coor = coor

        self.terminated = False
        self.velocity = (0, 0)
        s, _, _ = self._return()
        return s, self.terminated

    def step(self, a: Action) -> ((State, int, bool), list[Action]):
        """RETURN: state, reward, terminated"""
        if self.terminated:
            raise StopIteration("Call reset before step")
        if a not in self.action_space[(self.coor, self.velocity)]:
            raise ValueError("Wrong action!")

        # update velocity
        vr, vc = np.array(self.velocity) + np.array(a)
        self.velocity = vr, vc
        # check trace
        trace = self._moving_trace(self.coor)
        for coor in trace:
            if self._out_of_box(coor) or self._out_of_boundary(coor):
                self.reset()
                return self._return()
            elif self._finish(coor):
                self.coor = coor
                self.terminated = True
                return self._return()
        self.coor = coor
        return self._return()

    def _return(self):
        """RETURN: (state, reward, terminated), actions"""
        s = (self.coor, self.velocity)
        r = -1
        if self._finish(self.coor):
            r = 0
        return s, r, self.terminated

    def render(self):
        # Initialize an empty grid based on the shape attribute.
        grid = [['X'
                 for _ in range(self.shape[1])]
                for _ in range(self.shape[0])]

        # Fill the grid with special characters based on the coordinates.
        for r in range(self.track.shape[0]):
            for c in range(self.track.shape[1]):
                if (r, c) == self.coor:
                    grid[r][c] = "C"
                elif self.track[r, c] == 1:
                    grid[r][c] = "_"
        # Construct the string representation of the grid.
        outfile = StringIO()
        for row in grid:
            outfile.write(' '.join(row) + '\n')

        # Return the value of the outfile.
        print(outfile.getvalue())

    def _make_start_line(self) -> list[(int, int)]:
        r = self.shape[0] - 1  # last row
        res = [(r, c) for c in range(self.track.shape[1])
               if self.track[r, c] == 1]
        return res

    def _actions(self):
        return list(product(range(-1, 2, 1), range(-1, 2, 1)))

    def _velocities(self):
        return list(product(range(6), range(6)))

    def _make_spaces(self):
        """state: coordinate and velocity"""
        def valid_action_given_velocity(v):
            cont = []
            for a in self._actions():
                v1 = tuple(np.array(a) + np.array(v))
                if self._valid_velocity(v1):
                    cont.append(a)
            return tuple(cont)

        observations = list()
        action_space = dict()
        for r in range(self.track.shape[0]):
            for c in range(self.track.shape[1]):
                if (not self._out_of_boundary((r, c))
                        and not self._finish((r, c))):
                    if (r, c) in self._starting_line:
                        vs = self._velocities()
                    else:
                        vs = [v for v in self._velocities() if v != (0, 0)]
                    # observation
                    obs = [((r, c), v) for v in vs]
                    observations.extend(obs)
                    # action
                    for ob in obs:
                        vel = ob[1]
                        action_space[ob] = valid_action_given_velocity(vel)

        return tuple(observations), action_space

    def _make_finish_line(self) -> list[(int, int)]:
        c = self.shape[1] - 1  # last column
        res = [(r, c) for r in range(self.track.shape[0])
               if self.track[r, c] == 1]
        return res

    def _moving_trace(self, start) -> list[(int, int)]:
        """by velociaty"""
        r0, c0 = start
        vr = self.velocity[0]
        vc = self.velocity[1]
        if vr > vc:
            # move along row
            trace = [(r, round(c0 + (r0 - r) * vc / vr))
                     for r in range(r0 - 1, r0 - vr - 1, -1)]
        else:
            # move along column
            trace = [(round(r0 - (c - c0) * vr / vc), c)
                     for c in range(c0 + 1, c0 + vc + 1, 1)]
        return trace

    def _finish(self, coor) -> bool:
        if coor in self._finish_line:
            return True
        return False

    def _out_of_boundary(self, coor) -> bool:
        r, c = coor
        if self.track[r, c] == 0:
            return True
        return False

    def _out_of_box(self, coor) -> bool:
        try:
            self._coor2state(coor)
            return False
        except ValueError:  # outside of the boundary
            return True

    def _check_trace(self, start: (int, int)) -> ((int, int), int, bool):
        trace = self._moving_trace(start)
        for coor in trace:
            if self._out_of_box(coor):
                return (-1, -1), 0, True
            elif self._out_of_boundary(coor):
                return coor, 0, True
            elif self._finish(coor):
                return coor, 1, True
        return coor, 0, False

    def _valid_velocity(self, v):
        vr = v[0]
        vc = v[1]
        if (0 <= vr <= 5) and (0 <= vc <= 5) and not (vr == 0 and vc == 0):
            return True
        return False

    def _state2coor(self, s: int) -> (int, int):
        return np.unravel_index(s, self.shape)

    def _coor2state(self, coor: (int, int)) -> int:
        return np.ravel_multi_index(coor, self.shape)


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
    print("Observation space sample:", track.observation_space[: 10])
    print("Action space sample", track.action_space[((9, 0), (0, 0))])

    s, terminated = track.reset((9, 0))
    track.render()
    s, r, terminated = track.step((0, 1))
    track.render()
    s, r, terminated = track.step((1, 0))
    track.render()
    s, r, terminated = track.step((1, 0))
    track.render()
    s, r, terminated = track.step((1, 0))
    track.render()
    s, r, terminated = track.step((-1, 0))
    track.render()
    s, r, terminated = track.step((-1, 1))
    track.render()

    s, terminated = track.reset((9, 0))
    track.render()
    s, r, terminated = track.step((1, 0))
    track.render()
