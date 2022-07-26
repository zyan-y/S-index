

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")


save_path = r'./feature_selection/PLS/'


def vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros(p)
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips


def find_n_components(X, y):
    scores = []
    for num in range( 1, X.shape[1] ):
        score = []
        clf = make_pipeline(StandardScaler(), PLS(n_components=num))
        for seed in range(1000,1100):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            y_pred =  cross_val_predict(clf, X, y, cv=skf, n_jobs=5)
            s = roc_auc_score(y, y_pred)
            score.append(s)
        scores.append([ np.mean(score), num ])
    return max(scores)


def discard_n(X, y, n, dis_num, func, save=True):
    pls = PLS(n_components=n)
    pls.fit(X, y)

    if func == 'coef':
        rank_score = np.abs(pls.coef_[:,0])
    if func == 'vip':
        rank_score = vip(pls)
    sorted_ind = np.argsort(rank_score)[dis_num:]
    Xc = X[:,sorted_ind]
    if save == True:
        save_name = save_path+func+'_feature_'+str(sorted_ind.shape[0])
        np.savez(save_name, X=Xc, y=y)
    new_score = find_n_components(Xc, y)

    return Xc, new_score + [sorted_ind.shape[0]]


def discard_feature(X, y, dis_num, min_variable, func, save=True):
    record = [] #auc, n_comp, num of variables
    first_record = find_n_components(X, y)
    record.append( first_record + [X.shape[1]] )
    while X.shape[1] > min_variable:
        n = record[-1][1]
        X, score = discard_n(X, y, n, dis_num, func, save)
        record.append(score)
    if save == True:
        data = pd.DataFrame(record)
        header = ['auc', 'n_comp', 'n_variable']
        data.to_excel(r'./results/feature_selection_'+func+'_'+str(dis_num)+'.xlsx',header=header,index=False)
    return X


def find_discard(func, num, save=False):
    save_name = save_path+func+'_feature_'+str(num)+'.npz'
    # save_name = func+'_feature_'+str(num)+'.npz'
    data_sel = np.load(save_name)
    X_sel = data_sel['X']

    df = pd.read_excel(r'data.xlsx')
    header, data = df.columns.values[2:], df.values
    X, y = data[:,2:].astype('float'), data[:,1].astype('int')
    shuffle_ix = np.random.RandomState(777).permutation(np.arange(X.shape[0]))
    X, y = X[shuffle_ix], y[shuffle_ix]

    sel_idx = np.full((X_sel.shape[1], ), np.nan)
    for idx in range(X_sel.shape[1]):
        for i in range(X.shape[1]):
            if np.sum(X[:,i] - X_sel[:,idx]) == 0:
                sel_idx[idx] = i
                break
    dis_idx = np.setdiff1d(np.arange(X.shape[1]),sel_idx,True)
    print(func, num)
    print('Deleted features')
    print(header[dis_idx])
    print('Selected features')
    print(header[sel_idx.astype('int')])
    if save == True:
        path = r'./feature_selection/PLS/idx_feature_'+str(num)+'.npy'
        np.save(path, sel_idx.astype('int'))

if __name__ == '__main__':

    data = pd.read_excel(r'data.xlsx').values
    X, y = data[:,2:].astype('float'), data[:,1].astype('int')
    shuffle_ix = np.random.RandomState(777).permutation(np.arange(X.shape[0]))
    X, y = X[shuffle_ix], y[shuffle_ix]
    discard_feature(X, y, 5, 20, 'vip')


