
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from xgboost import XGBClassifier as XGB
import shap

color = plt.get_cmap('RdYlBu_r')(np.linspace(0.1, 0.9, 4))

def generate_shap_value():
    header = pd.read_excel('data.xlsx').columns.values[2:]
    data = pd.read_excel('data.xlsx').values[:,2:].astype('float')
    idx = np.load(r'feature_idx.npy')
    data, header = data[:,idx], header[idx]

    model = XGB()
    model.load_model('model.json')
    explainer = shap.TreeExplainer(model)
    shap_values = shap.TreeExplainer(model).shap_values(data)
    shap_interaction_values = explainer.shap_interaction_values(data)
    np.save(r'./result/shap_values.npy', shap_values)
    np.save(r'./result/shap_interaction_values.npy', shap_interaction_values)


def plot_one(feature, value, name):
    fig = plt.figure(figsize=(4,3))
    fig.patch.set_alpha(0.)
    plt.rc('font',family='Arial')
    plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})
    plt.scatter(feature, value, c=color[3], marker='.', s=20)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.xlim((0,1))
    plt.ylabel('Effect Value')
    plt.xlabel(name)
    plt.tight_layout()
    plt.show()


def plot_interaction(feature1, feature2, value, name1, name2):
    print(name1, name2, min(value), max(value))
    
    fig = plt.figure(figsize=(4,3))
    fig.patch.set_alpha(0.)
    plt.rc('font',family='Arial')
    plt.rcParams.update({'font.size': 12, 'legend.fontsize': 12})
    plt.scatter(feature1, feature2, c=plt.get_cmap('RdYlBu_r')(value), marker='.', s=20)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylim((0,1))
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.tight_layout()
    plt.show()


def explain_dataset():
    header = pd.read_excel('data.xlsx').columns.values[2:]
    data = pd.read_excel('data.xlsx').values[:,2:].astype('float')
    idx = np.load(r'feature_idx.npy')
    data = data[:,idx]
    header = header[idx]

    shap_values = np.load(r'./result/shap_values.npy')

    i = 2
    plot_one(data[:,i], shap_values[:,i], header[i])

    shap_interaction_values = np.load(r'./result/shap_interaction_values.npy')
    a, b = 3, 5
    plot_interaction(data[:,a], data[:,b], shap_interaction_values[:,b,a]*2, header[a], header[b])


def find_key():
    path = r'./chr_score//'
    names = ['Athaliana_chr2', 'Celegans_chrII','DNA_storage','Drosophila_chrY',\
                'Human_chr22', 'JCVI_syn1.0', 'Oryza_sativa_chr9', \
                'Yeast_synV', 'Yeast_synX', 'Mouse_chr19', 'CETH_2.0']
    idx = np.load(r'feature_idx.npy')
    header = pd.read_excel('data.xlsx').columns.values[2:][idx].tolist()
    
    model = XGB()
    model.load_model('model.json')
    
    record = []
    for name in names:
        save_name = path + name + '.npy'
        data = np.load(save_name)[:,idx]
        y_prob = model.predict_proba(data)[:,1]

        shap_values = shap.TreeExplainer(model).shap_values(data)
        rank_shap = np.argmax(shap_values, axis=1)
        rank_shap = rank_shap[ y_prob>0.5 ]
        count = rank_shap.shape[0]
        key_factors = [  np.count_nonzero(rank_shap == i)/count for i in range(len(idx)) ]
        record.append( [name] + key_factors + [count])
    
    record = pd.DataFrame(record)
    header = ['name'] + header + ['count']
    record.to_excel(r'./result/key_factors.xlsx', header=header, index=None)



if __name__ == '__main__':
    explain_dataset()
    find_key()

