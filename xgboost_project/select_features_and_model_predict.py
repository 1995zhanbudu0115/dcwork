# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:55:25 2020

@author: zhanb
@project: 特征筛选与模型训练
"""
import pandas as pd
import numpy as np
from scipy import sparse
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve,auc
import pickle
import os
import matplotlib.pyplot as plt

save_path = './data'
train_data = pd.read_csv('./data/train/features.csv')
test_data = pd.read_csv('./data/test/features.csv')
x_train = train_data.iloc[:,2:]
y_train = train_data.iloc[:,1]
x_test = test_data.iloc[:,1:]


# 交叉验证
def kfold_test(k,xtr,ytr,save_name='kfold_models'):
    print('开始{}折交叉验证...'.format(k))
    roc_scores = []
    models = []
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    i = 1
    for train_index, val_index in kf.split(xtr, ytr):
        print('{}折训练中...'.format(i))
        xt,xv = xtr[train_index], xtr[val_index]
        yt, yv = ytr[train_index], ytr[val_index]
        eval_set = [(xt, yt), (xv, yv)]
        model = XGBClassifier(n_estimators=300,n_jobs = 6,random_state=42)
    
        model.fit(xt, yt, eval_metric=['auc', "error", "logloss"], 
                  eval_set=eval_set, verbose=False)
        score = model.predict_proba(xv)[:,1]
        roc = roc_auc_score(yv,score)
        roc_scores.append(roc)
        models.append(model)
        i+=1
    print('mean auc: ',np.mean(roc_scores))
    with open(save_path+'/{}.pickle'.format(save_name),'wb') as f:
        pickle.dump(models,f)
    return models

# 筛选特征
def select_features(xtr,ytr):
    kfold_path = save_path+'/kfold_models.pickle'
    if os.path.exists(kfold_path):
        kfold_models = pickle.load(open(kfold_path,'rb'))
    else:
        kfold_models = kfold_test(10,xtr,ytr)
    sele_features=set()
    for m in kfold_models:
        fea_df = pd.DataFrame(m.feature_importances_,index=list(x_train),columns=['score'])
        fea_df = fea_df.sort_values(by=['score'],ascending=False)
        # 每次循环取并集
        sele_features |= set(fea_df.loc[fea_df['score']>0].index)
    pickle.dump(sele_features,open(save_path+'\selected_features.pickle','wb'))
    return list(sele_features)


# 验证筛选特征后的模型是否稳定
def validate_after_selected(xtr,ytr):
    print('进行模型验证...')
    vaildate_models = kfold_test(10,xtr,ytr,save_name='validate_models')

# 模型预测主程序
def model_predict(ids,xtr,ytr,xte):
    x_tr_csr = sparse.csr_matrix(xtr)
    if os.path.exists(save_path+'\selected_features.pickle'):
        selected_features = pickle.load(open(save_path+'\selected_features.pickle','rb'))
    else:
        selected_features = select_features(x_tr_csr,ytr)
    x_tr_sele = xtr[selected_features]
    x_tr_csr_sele = sparse.csr_matrix(x_tr_sele)
    x_te_sele = xte[selected_features]
    x_te_csr_sele = sparse.csr_matrix(x_te_sele)
#    不进行模型验证时可以将该行注释掉
#    validate_after_selected(x_tr_csr_sele,ytr)
    xgc = XGBClassifier(n_estimators=300,n_jobs = 6,random_state=42)
    xgc.fit(x_tr_csr_sele,ytr)
    yscore = xgc.predict_proba(x_te_csr_sele)[:,1]
    predict_df = pd.DataFrame(np.array([ids,yscore]).T,columns=['id','score'])
    predict_df.to_csv('score_set.txt',sep='\t',index=False)
    print('提交数据保存完毕')
    
if __name__ == '__main__':
    model_predict(list(test_data['id']),x_train,y_train,x_test)
    
    

    
    