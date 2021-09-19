# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:02:30 2020

@author: duxx
"""
#%%
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import roc_curve
import seaborn as sns

#%%
# --------------------------原始原理方法-----------------------------------
x,y = make_classification(n_samples=6000,n_features=50,random_state=123)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=123)

### 第一层模型
clfs = [GBDT(n_estimators=100),RF(n_estimators=100),
       ET(n_estimators=100),ADA(n_estimators=100)]
#%%
X_train_stack  = np.zeros((X_train.shape[0], len(clfs)))
X_test_stack = np.zeros((X_test.shape[0], len(clfs))) 

### 6折stacking
n_folds = 6
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=123)
for i,clf in enumerate(clfs):
    print("分类器：{}".format(clf))
    X_stack_test_n = np.zeros((X_test.shape[0], n_folds))
    for j,(train_index,test_index) in enumerate(skf.split(X_train,y_train)):
                print(f'第{j}折')
                tr_x = X_train[train_index]
                tr_y = y_train[train_index]
                clf.fit(tr_x, tr_y)
                #生成stacking训练数据集
                X_train_stack[test_index, i] = clf.predict_proba(X_train[test_index])[:,1]
                X_stack_test_n[:,j] = clf.predict_proba(X_test)[:,1]
    #生成stacking测试数据集
    X_test_stack[:,i] = X_stack_test_n.mean(axis=1) 
    
###第二层模型LR
clf_second = LogisticRegression(solver="lbfgs")
clf_second.fit(X_train_stack,y_train)
pred = clf_second.predict_proba(X_test_stack)[:,1]
roc_auc_score(y_test,pred)#0.9856418918918919
tpr,fpr,_ = roc_curve(y_test,pred)
sns.set_style('darkgrid')
sns.lineplot(x=tpr,y=fpr)

#%%
# --------------------------直接调包方法sklaern-----------------------------------

from sklearn.ensemble import StackingClassifier

clfs2 = [('gbdt',GBDT(n_estimators=100)),('rf',RF(n_estimators=100)),
         ('et',ET(n_estimators=100)),('ada',ADA(n_estimators=100))]
skf = StackingClassifier(estimators=clfs2,final_estimator=LogisticRegression(),cv=3)
skf.fit(X_train,y_train)
pred = skf.predict_proba(X_test)[:,1]
roc_auc_score(y_test,pred) #0.985702347083926
import scorecardpy as sc
sc.perf_eva(y_test, pred)
# --------------------------直接调包方法mlxtend-----------------------------------
#%%
from mlxtend.classifier import StackingClassifier as mStackingClassifier

sclf = mStackingClassifier(classifiers=clfs,use_probas=True,
                              # use_probas=True, 类别概率值作为meta-classfier的输入
                              # average_probas=False,  是否对每一个类别产生的概率值做平均
                              meta_classifier=LogisticRegression())

sclf.fit(X_train, y_train)
#%%
# clfs.append(sclf)
for clf, label in zip(clfs, ['gbdt', 'rf', 'et', 'ada','StackingModel']):
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
        print("roc_auc: %0.4f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
pred_y = sclf.predict_proba(X_test)[:,1]
roc_auc_score(y_test,pred_y)
#%%
from lightgbm import LGBMClassifier
clfs = [ADA(n_estimators=100)]
sclf = mStackingClassifier(classifiers=clfs,use_probas=True,
                              # use_probas=True, 类别概率值作为meta-classfier的输入
                              # average_probas=False,  是否对每一个类别产生的概率值做平均
                              meta_classifier=LogisticRegression())

sclf.fit(X_train, y_train)
pred_y = sclf.predict_proba(X_test)[:,1]
roc_auc_score(y_test,pred_y)

#%%
# lgbc = LGBMClassifier()
# lgbc.fit(X_train,y_train)
# pred_y = lgbc.predict_proba(X_test)[:,1]
ada = ADA(n_estimators=100)
ada.fit(X_train,y_train)
pred_y = ada.predict_proba(X_test)[:,1]
roc_auc_score(y_test,pred_y)
# %%
