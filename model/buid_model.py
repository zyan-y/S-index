
import pandas as pd
import numpy as np
from xgboost import XGBClassifier as XGB

if __name__ == '__main__':
    data = pd.read_excel('data.xlsx').values
    X, y = data[:,2:].astype('float'), data[:,1].astype('int')
    idx = np.load(r'feature_idx.npy')
    X = X[:,idx]

    params = { 'n_estimators': 500, 'max_depth':5, 'learning_rate':0.1}
    model = XGB(use_label_encoder=False, eval_metric='logloss', **params)
    
    model.fit(X, y)
    model.save_model('model.json')
