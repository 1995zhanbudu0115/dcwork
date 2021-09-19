#%%
import pandas as pd
import pickle
import scorecardpy as sc

#%%
tra_x = pd.read_pickle('data/sxd_b_tra_x.pkl')
oot_x = pd.read_pickle('data/sxd_b_oot_x.pkl')
clf = pickle.load(open('data/sxd_b_clf.pkl','rb+'))
bins_dic = pickle.load(open('data/sxd_b_bins_dic.pkl','rb+'))
# %%
card = sc.scorecard(bins_dic,clf,tra_x.columns)
tra_score = sc.scorecard_ply(tra_x, card, only_total_score=False)
oot_score = sc.scorecard_ply(oot_x, card, only_total_score=False)
pass