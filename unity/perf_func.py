from sklearn.metrics import roc_curve
from unity import init_pkg as ini
from sklearn.metrics import roc_auc_score,roc_curve

class PertFunc:

    @staticmethod   
    def eva_plot(y_true,y_pred):
        fpr,tpr,_ = roc_curve(y_true,y_pred)
        auc = roc_auc_score(y_true,y_pred)
        f,axs = ini.plt.subplots(1,2,figsize=(12,5))
        PertFunc.ks_plot(fpr,tpr,ax=axs[0])
        PertFunc.roc_plot(fpr,tpr,auc,ax=axs[1])
        
    # KS简化版
    @staticmethod
    def ks_plot(fpr,tpr,**kwargs):
        ks_array = abs(tpr-fpr)
        ks_index = fpr[ks_array==max(ks_array)]
        ks = max(tpr-fpr)
        if kwargs['ax']:
            ax = kwargs['ax']
            ax.plot(fpr,tpr,'k-')
            ax.plot(fpr,tpr-fpr,'b-')
            ax.plot([0,1],'k-')
            ax.plot([ks_index,ks_index],[0,ks],'r--')
            ax.text(ks_index,ks,'KS:'+str(round(ks,4)), horizontalalignment='center', color='b')
        else:
            ini.plt.plot(fpr,tpr,'k-')
            ini.plt.plot(fpr,tpr-fpr,'b-')
            ini.plt.plot([0,1],'k-')
            ini.plt.plot([ks_index,ks_index],[0,max(tpr-fpr)],'r--')

    # ROC简化版 
    @staticmethod    
    def roc_plot(fpr,tpr,auc,title='',**kwargs):
        if kwargs['ax']:
            ax = kwargs['ax']
            ax.plot(fpr,tpr,'k-')
            ax.fill_between(fpr, 0, tpr, color='blue', alpha=0.1)
            ax.plot([0,1],'r--')
            ax.set(title=title+'ROC',
              xlabel='FPR', ylabel='TPR', 
              xlim=[0,1], ylim=[0,1], aspect='equal')
            ax.text(0.55,0.45, 'AUC:'+str(round(auc,4)), horizontalalignment='center', color='b')
        else:
            ini.plt.plot(fpr,tpr,'k-')
            ini.plt.fill_between(fpr, 0, tpr, color='blue', alpha=0.1)
            ini.plt.plot([0,1],'r--')
            ini.plt.gca().set(title=title+'ROC',
              xlabel='FPR', ylabel='TPR', 
              xlim=[0,1], ylim=[0,1], aspect='equal')
            ini.plt.text(0.55,0.45, 'AUC:'+str(round(auc,4)), horizontalalignment='center', color='b')
