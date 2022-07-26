
from cProfile import label
from email import header
from turtle import width
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np



def plot_bar(name, cut):
    path = r'./chr_score//'
    data = pd.read_excel( path + name + '_result.xlsx',header=None).values
    idx = [ (i+0.5)*cut for i in range(data.shape[0])]
    
    c = {0:'#4574B3', 1:'#D62F27'}
    c_idx = np.around(data[:,1])
    colors = [c[i] for i in c_idx]

    fig = plt.figure(figsize=(5,3))
    fig.patch.set_alpha(0.)
    plt.rc('font',family='Arial')
    plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})

    plt.bar(idx, data[:,1], color=colors, width=cut, lw=0.25, edgecolor='black')
    plt.axhline(0.5, linestyle='--', c='black', lw=0.5)

    plt.ylim((0,1))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Chromosome Location (kb)')
    plt.ylabel('Difficulty Score')
    plt.tight_layout()
    plt.show()


def plot_heat(name):
    path = r'chr_score//'
    data = pd.read_excel( path + name+'_result_.xlsx',header=None).values.ravel().tolist()

    idx = [i*2 for i in range(len(data))]
    
    c = {0:'#4574B3', 1:'#D62F27'}
    c_idx = np.around(data)

    fig = plt.figure(figsize=(8,0.6))
    plt.rcParams['axes.facecolor'] = 'grey'
    for i in range(len(idx)):
        if c_idx[i]:
            plt.axvline(idx[i], linestyle='-', c=c[c_idx[i]], linewidth=0.02) #, linewidth=10

    plt.yticks([])
    plt.xticks([])
    plt.xlim((-2, idx[-1]+2))
    plt.tight_layout()
    savepath = r'./chr_analysis/chr_loc//'
    plt.savefig(savepath + name +'.png')
    plt.savefig(savepath + name +'.svg')
    plt.close()
    print('finish '+name)


def plot_cum_bar():
    filename = r'results/chr_results.xlsx'
    filename = r'results/key_factors.xlsx'
    df = pd.read_excel(filename, sheet_name=0, header=0, usecols=list(range(0,7))).values
    name, data = df[:,0].astype('str'), df[:,1:].astype('float')
    data = data * 100

    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlBu_r')(np.linspace(0.1, 0.9, 10))

    fig = plt.figure(figsize=(10,4))
    fig.patch.set_alpha(0.)
    plt.rc('font',family='Arial')
    plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})
    plt.tick_params(bottom=False)

    for i in range(data.shape[1]):
        heights = data[:, i] # 取第一列数值
        starts = data_cum[:, i] - heights # 取每段的起始点
        plt.bar(name, heights, bottom=starts, width=0.4, color=category_colors[i])

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.2), ncol=3)
    plt.ylim((0,100))
    plt.ylabel('Ratio of fragments (%)')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    names = ['Athaliana_chr2', 'Celegans_chrII','DNA_storage','Drosophila_chrY',\
                'Human_chr22', 'JCVI_syn1.0', 'Oryza_sativa_chr9', \
                'Yeast_synV', 'Yeast_synX', 'Mouse_chr19', 'CETH_2.0']
    [plot_heat(name) for name in names]
    plot_bar()
    plot_cum_bar()