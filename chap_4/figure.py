# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:16:50 2024

@author: ccw
"""
import numpy as np
import matplotlib.pyplot as plt
from gambler import Casino, Gambler


def figure_4_3(ph=0.4):

    def dict2plot_data(dict_):
        x, y = [np.zeros(len(dict_)) for _ in range(2)]
        for i, (s, v) in enumerate(dict_.items()):
            x[i] = s
            y[i] = v
        return x, y

    figure, axs = plt.subplots(nrows=2, ncols=1)

    agent = Gambler(Casino(ph=ph))
    # first plot
    ax = axs[0]
    for _ in range(3):
        agent.value_update()
        x, y = dict2plot_data(agent.get_values())
        ax.plot(x, y)
    for _ in range(29):
        agent.value_update()
    x, y = dict2plot_data(agent.get_values())
    ax.plot(x, y)
    agent.value_iteration()
    x, y = dict2plot_data(agent.get_values())
    ax.plot(x, y)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks([1, 25, 50, 75, 99])

    # second plot
    ax = axs[1]
    x, y = dict2plot_data(agent.get_policy())
    ax.plot(x, y)
    ax.set_xticks([1, 25, 50, 75, 99])
    ax.set_yticks([1, 10, 20, 30, 40, 50])


if __name__ == '__main__':
    figure_4_3()
    figure_4_3(0.25)
    figure_4_3(0.55)
