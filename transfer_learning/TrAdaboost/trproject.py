# %%
import pandas as pd
import TrAdaboost as tra
from TrAaboostOrg import TrAdaboostClassifier
from sklearn.datasets import make_classification
import numpy as np
import scorecardpy as sc
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# %%
alldat = pd.read_csv('../dataset/base_v3_new.csv')
alldat = alldat.drop(['19Q4日均余额','mon_aum_bal_M1','aum_bal_M1','aum_bal_M1','mon_aum_bal_M2','aum_bal_M2',
                      'mon_aum_bal_M3','aum_bal_m3','decrease1','decrease2','decrease2'],axis=1)

# %%
tdx = alldat.loc[alldat['客户等级'].isin([0]),~alldat.columns.str.contains('label')]
tdy = alldat.loc[alldat['客户等级'].isin([0]),'label']
tsx = alldat.loc[alldat['客户等级'].isin([2,3]),~alldat.columns.str.contains('label')]
tsy = alldat.loc[alldat['客户等级'].isin([2,3]),'label']
ts_train_x,ts_valid_x,ts_train_y,ts_valid_y = train_test_split(tsx,tsy,test_size=0.3,random_state=0,shuffle=True)
# %%
# baseline
# params = {'n_estimators':80,'reg_alpha':2,'reg_lambda':10,'max_depth':3,'min_child_weight':10}
bse_train_x = np.concatenate([tdx,ts_train_x],axis=0)
bse_train_y = np.concatenate([tdy,ts_train_y],axis=0)
# lgbc = LGBMClassifier(**params)
ada = AdaBoostClassifier(base_estimator=LogisticRegression(),n_estimators=20,random_state=0,algorithm='SAMME.R')
ada.fit(bse_train_x,bse_train_y)
ts_valid_pred = ada.predict_proba(ts_valid_x)[:,1]
ts_train_pred = ada.predict_proba(ts_train_x)[:,1]
bse_train_pred = ada.predict_proba(bse_train_x)[:,1]
print(f'baseline_auc: {roc_auc_score(ts_valid_y,ts_valid_pred)}')
fpr,tpr,_ = roc_curve(ts_valid_y,ts_valid_pred)
print(f'baseline_ks: {max(tpr-fpr)}')
# %%
# trb = tra.TradaBoostClassifier(epoch=10,learner=LGBMClassifier(**params))
# trb.fit(tsx.values,tdx.values,tsy.values,tdy.values,ts_valid_x.values)

trb = TrAdaboostClassifier(base_classifier=LogisticRegression(),N=20)
trb.fit(tdx,ts_train_x,tdy,ts_train_y)
# %%
ts_pred = trb.predict_proba(ts_valid_x)
print(f'{roc_auc_score(ts_valid_y, ts_pred)}')
# %%
final_clf = trb.classifiers[-1]
ts_pred = final_clf.predict_proba(ts_valid_x)[:,1]
roc_auc_score(ts_valid_y,ts_pred)
# %%
