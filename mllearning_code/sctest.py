import scorecardpy as sc

# dat = sc.germancredit()
    
# # filter variable via missing rate, iv, identical value rate
# dt_sel = sc.var_filter(dat, "creditability")

# # woe binning ------
# bins = sc.woebin(dt_sel, "creditability")
# dt_woe = sc.woebin_ply(dt_sel, bins)

# y = dt_woe.loc[:,'creditability']
# X = dt_woe.loc[:,dt_woe.columns != 'creditability']

# # logistic regression ------
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(penalty='l1', C=0.9, solver='saga')
# lr.fit(X, y)

# # predicted proability
# dt_pred = lr.predict_proba(X)[:,1]

# # performace ------
# # Example I # only ks & auc values
# sc.perf_eva(y, dt_pred, show_plot=False)
# pass

import pandas as pd

tra_x = pd.read