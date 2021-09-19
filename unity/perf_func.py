from sklearn.metrics import roc_curve
from unity import init_pkg as ini

# KS简化版
def ks_plot(fpr,tpr):
    ini.plt.figure(figsize=(6,6))
    ks_array = tpr-fpr
    ks_index = fpr[ks_array==max(ks_array)]
    ini.plt.plot(fpr,tpr,'k-')
    ini.plt.plot(fpr,tpr-fpr,'b-')
    ini.plt.plot([0,1],'k-')h 
    ini.plt.plot([ks_index,ks_index],[0,max(tpr-fpr)],'r--')
    
# ROC简化版    
def roc_plot(fpr,tpr):
    ini.plt.plot(fpr,tpr,'k-')
