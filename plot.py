import matplotlib as mpl
import numpy as np
import pickle
from scipy.stats import kde


mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'font.size': 12,
    'text.usetex': False,
})


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


def plot_dep_k(data, fpath=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    plt.subplots_adjust(wspace=0.3)

    ax1.set_xlabel('Number of selected items (K)')
    ax1.set_ylabel('Absolute error for maximum')

    ax2.set_xlabel('Number of selected items (K)')
    ax2.set_ylabel('Absolute error for minimum')

    ax3.set_xlabel('Number of selected items (K)')
    ax3.set_ylabel('Calculation time (sec.)')

    for name, func in data.items():
        k = list(func.keys())
        t = [item['t'] for item in func.values()]
        e_min = [item['e_min'] for item in func.values()]
        e_max = [item['e_max'] for item in func.values()]

        ax1.plot(k, e_min, label=name,
            marker='o', markersize=8, linewidth=0)
        ax2.plot(k, e_max, label=name,
            marker='o', markersize=8, linewidth=0)
        ax3.plot(k, t, label=name,
            marker='o', markersize=8, linewidth=0)

    prep_ax(ax1, xlog=False, ylog=True)
    prep_ax(ax2, xlog=False, ylog=True)
    prep_ax(ax3, xlog=False, ylog=True, leg=True)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def plot_random_small_hist_dens(data_all, fpath=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colors = ['green', 'blue', 'orange']
    alphas = [0.7, 0.85, 0.95]
    bins = [15, 15, 30]

    for i, k in enumerate(data_all.keys()):
        v = np.array(data_all[k]['e_max'])

        density = kde.gaussian_kde(v, bw_method=0.3)
        x = np.linspace(0.7, 1., 1000)
        y = density(x)

        ax.plot(x, y, label=f'K = {k}', color=colors[i])

    prep_ax(ax, xlog=False, ylog=True, leg=True)
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Density')

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def plot_random_small_hist(data_all, fpath=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colors = ['green', 'blue', 'orange']
    alphas = [0.9, 0.5, 0.5]
    bins = [100, 30, 30]
    aligns = ['left', 'mid', 'right']

    for i, k in enumerate(data_all.keys()):
        v = np.array(data_all[k]['e_max'])
        #w = np.ones_like(v) / float(len(v))

        #ax.hist(v, bins[i], label=f'K = {k}', density=False, log=True,
        #    facecolor=colors[i], alpha=alphas[i], weights=w)

        #minv = np.min(v)
        #bins_i = int(10/minv)
        bins_i = np.linspace(0.5, 1, 40)
        ax.hist(v, bins_i, label=f'K = {k}', density=True, log=False,
            facecolor=colors[i], alpha=alphas[i], align=aligns[i])

    prep_ax(ax, leg=True)
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Density')

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def plot_dep_random_k(data_all, fpath=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    plt.subplots_adjust(wspace=0.45)

    ax1.set_xlabel('Number of selected items (K)')
    ax1.set_ylabel('Absolute error for maximum')

    ax2.set_xlabel('Number of selected items (K)')
    ax2.set_ylabel('Absolute error for minimum')

    ax3.set_xlabel('Number of selected items (K)')
    ax3.set_ylabel('Calculation time (sec.)')

    colors = ['green', 'blue', 'orange']

    for i, (d, data) in enumerate(data_all.items()):
        k = list(data.keys())
        t = [item['t'] for item in data.values()]
        e_min = np.array([item['e_min'] for item in data.values()])
        e_max = np.array([item['e_max'] for item in data.values()])
        e_min_var = np.array([item['e_min_var'] for item in data.values()])
        e_max_var = np.array([item['e_max_var'] for item in data.values()])
        label = f'{d}-dim'

        ax1.plot(k, e_max, label=label,
            color=colors[i], marker='o', markersize=8, linewidth=2)
        ax1.fill_between(k, e_max - e_max_var, e_max + e_max_var,
            color=colors[i], alpha=0.1)

        ax2.plot(k, e_min, label=label,
            color=colors[i], marker='o', markersize=8, linewidth=2)
        ax2.fill_between(k, e_min - e_min_var, e_min + e_min_var,
            color=colors[i], alpha=0.1)

        ax3.plot(k, t, label=label,
            color=colors[i], marker='o', markersize=8, linewidth=2)

    prep_ax(ax1, xlog=True, ylog=False)
    prep_ax(ax2, xlog=True, ylog=False)
    prep_ax(ax3, xlog=True, ylog=False, leg=True)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None):
    if xlog:
        ax.semilogx()
    if ylog:
        ax.semilogy()

    if leg:
        ax.legend(loc='best', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)
