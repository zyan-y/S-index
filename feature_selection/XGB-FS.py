


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier as XGB
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")


def cv(X, y):
    scores = []
    model = XGB(use_label_encoder=False,eval_metric='logloss')
    clf = make_pipeline(StandardScaler(), model)
    for seed in range(100):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        score = cross_val_score(clf, X, y, cv=skf, n_jobs=5, scoring='roc_auc')
        scores.append(score)
    return np.mean(scores)


def discard_n(X, y, dis_num=1, save=True):
    xgb = XGB(use_label_encoder=False,eval_metric='logloss')
    
    xgb.fit(X, y)
    rank_score = xgb.feature_importances_
    sorted_ind = np.argsort(rank_score)[dis_num:]
    Xc = X[:,sorted_ind]
    if save == True:
        save_name = r'./feature_selection/XGB/feature_'+str(sorted_ind.shape[0])
        np.savez(save_name, X=Xc, y=y)
    new_score = cv(Xc, y)
    print(rank_score[sorted_ind[0]], new_score, sorted_ind.shape[0])
    return Xc, [sorted_ind.shape[0], new_score]


def discard_feature(X, y, dis_num, min_variable, save=True):
    record = [] #auc, num of variables
    first_record = cv(X, y)
    record.append( [X.shape[1], first_record] )
    while X.shape[1] > min_variable:
        X, score = discard_n(X, y, dis_num, save)
        record.append(score)
    if save == True:
        data = pd.DataFrame(record)
        header = ['n_variable', 'auc']
        data.to_excel(r'./results/feature_selection_XGB.xlsx',header=header,index=False)
    return X


def feature_selection(num):
    load_name = r'./feature_selection/PLS/vip_feature_'+str(num)+'.npz'
    data_sel = np.load(load_name)
    X, y = data_sel['X'], data_sel['y']
    discard_feature(X, y, 1, 2, True)


def find_discard(num, save=False):
    save_name = r'./feature_selection/XGB_AUC/feature_'+str(num)+'.npz'
    data_sel = np.load(save_name)
    X_sel = data_sel['X']

    df = pd.read_excel(r'data.xlsx')
    header, data = df.columns.values[2:], df.values
    X = data[:,2:].astype('float')
    shuffle_ix = np.random.RandomState(777).permutation(np.arange(X.shape[0]))
    X = X[shuffle_ix]

    sel_idx = np.full((X_sel.shape[1], ), np.nan)
    for idx in range(X_sel.shape[1]):
        for i in range(X.shape[1]):
            if np.sum(X[:,i] - X_sel[:,idx]) == 0:
                sel_idx[idx] = i
                break
    dis_idx = np.setdiff1d(np.arange(X.shape[1]),sel_idx,True)
    print('Deleted features')
    print(header[dis_idx])
    print('Selected features')
    print(header[sel_idx.astype('int')])
    if save == True:
        path = r'./feature_selection/XGB/idx_feature_'+str(num)+'.npy'
        np.save(path, sel_idx.astype('int'))


if __name__ == '__main__':
    feature_selection(28)
    # find_discard(6)