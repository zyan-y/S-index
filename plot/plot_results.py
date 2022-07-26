


import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


color =  ['#D62F27', 'tab:orange', '#4574B3']


def plot_fs_PLS():
    data = pd.read_excel(r'./results/feature_selection_PLS.xlsx').values[:,2:]
    fig = plt.figure(figsize=(4.5,3))
    fig.patch.set_alpha(0.)
    plt.rc('font',family='Arial')
    plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})
    plt.plot(data[:,0], data[:,1], '-', color=color[0], marker='.', lw=1, markersize=4)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.show()


def plot_fs_XGB():
    data = pd.read_excel(r'./results/feature_selection_XGB.xlsx').values
    fig = plt.figure(figsize=(4.5,3))
    fig.patch.set_alpha(0.)
    plt.rc('font',family='Arial')
    plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})
    plt.plot(data[:,0], data[:,1], '-', color=color[0], marker='.', lw=1, markersize=4)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.show()


def plot_fs_result():
    df = pd.read_excel(r'./results/compare_feature_selection.xlsx', sheet_name=0)
    metrics, names, data = df.columns, df.values[:,0].astype('str'), df.values[:,1:].astype('float')
    data = np.around(data,3)
    data, names = data[0:3], names[0:3]

    fig = plt.figure(figsize=(5,2.5))
    fig.patch.set_alpha(0.)
    plt.tick_params(bottom=False)
    plt.rc('font',family='Arial')
    plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})
    a = int(len(metrics) / 2)
    x = np.arange(a)+1
    w, gap = 0.16, 0.04
    error_params = dict(elinewidth=1, ecolor='black', capsize=5)
    
    plt.bar(x-w-gap, data[0][0:a], width=w, lw=0, color=color[0], label=names[0],yerr=data[0][a:],error_kw=error_params)
    plt.bar(x, data[1][0:a], width=w, lw=0, color=color[1], label=names[1],yerr=data[1][a:],error_kw=error_params)
    plt.bar(x+w+gap, data[2][0:a], width=w, lw=0, color=color[2], label=names[2],yerr=data[2][a:],error_kw=error_params)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1))
    ax.set_xticklabels(metrics)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim((0,1))
    plt.legend(frameon=False, loc=9, fontsize=8, borderaxespad=0)
    plt.tight_layout()
    plt.show()


def plot_fi_pie():
    data = pd.read_excel(r'./results/feature_importance.xlsx', header=None).values
    labels = data[:,0].astype('str')
    datas = data[:,1].astype('float')
    idx = np.argsort(datas)
    datas, labels = datas[idx], labels[idx]
    
    fig = plt.figure(figsize=(5,3))
    fig.patch.set_alpha(0.)
    plt.rc('font',family='Arial')
    plt.rcParams.update({'font.size': 10, 'legend.fontsize': 10})
    plt.tick_params(left=False)
    
    category_colors = plt.get_cmap('RdYlBu_r')(np.linspace(0.7, 0.9, 6))
    explode = [0.01]*5 + [0.2]
    plt.pie(datas, explode=explode, labels=labels, autopct='%1.1f%%', colors=category_colors)
    plt.show()


def plot_swarm():
    data = pd.read_excel('data.xlsx')
    plt.figure(figsize=(4,3))
    x, y = 'Labels', '100-bp min IE'
    sns.boxplot(x, y, data=data, boxprops={'facecolor':'None'}, showfliers=False, showmeans=False)
    sns.swarmplot(x, y, data = data, s=2)
    plt.tight_layout()
    plt.show()
