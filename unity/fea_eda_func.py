import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy

#定义异常
class paraException(BaseException):
    def __init__(self,mesg="raise a myException"):
        print(mesg)
        
class FeaEdaFunc:
        
    @staticmethod
    def build_cert_refer_feas(df,cert_cols,curr_year=2020):
        """
        根据身份证号生成基础衍生特征:
        身份证户籍地址
        身份证性别
        身份证年龄
        """
        df['cert_sex'] = df[cert_cols].str.get(16).astype(int) % 2
        df['cert_resid_prov'] = df[cert_cols].str.slice(0,7).astype(int) // 100000 * 10000
        df['cert_age'] = df[cert_cols].str.slice(6,15)
        df['cert_age'] = 2020-df[cert_cols].str.slice(6,10).astype(int)
        df.drop(cert_cols,axis=1,inplace=True)
        return df
    
    @staticmethod
    def build_intect_feas():
        pass
    
    @staticmethod
    def bin_group_single(df, col, target):
        """
        单个特征iv,woe
        """
        regroup = df.groupby([col])['tag'].agg(['count','sum']).replace(0,0.0001)\
                    .rename({'count':'total','sum':'bad'},axis=1)\
                    .assign(variable=col,
                            count_distr = lambda x:x['total']/sum(x['total']),
                            good=lambda x:x['total']-x['bad'],
                            badprob = lambda x:x['bad']/x['total'],
                            DistrBad = lambda x:x['good']/sum(x['good']),
                            DistrGood = lambda x:x['bad']/sum(x['bad']),
                            woe = lambda x:np.log(x['DistrBad']/x['DistrGood']),
                            bin_iv = lambda x:(x['DistrBad']-x['DistrGood'])*x['woe'],
                            total_iv = lambda x:sum(x.bin_iv))
        regroup.reset_index(level=0,inplace=True)
        return regroup
    
    @staticmethod
    def bin_group(df, cols, target,missing=True):
        """
        """
        bin_df_dicts = {}
        tdf = df.copy()
        if missing: tdf.loc[:,cols] = tdf.fillna('missing')
        for c in cols:
            bin_df = FeaEdaFunc.bin_group_single(tdf,c,target)
            bin_df_dicts[c] = bin_df
        return bin_df_dicts
    
    @staticmethod
    def badrate_ply(df,cols, bin_df_dicts,missing=True):
        """
        badrate编码转化
        """
        tdf = df.copy()
        if missing: tdf.loc[:,cols] = tdf.fillna('missing')
        bin_badp_dicts = {c:dict(zip(bin_df_dicts[c][c],bin_df_dicts[c]['badprob'])) for c in cols}
        df.replace(bin_badp_dicts,inplace=True)
        return tdf
        
    @staticmethod
    def __check_monotony(x):
        """
        查看单调性
        """
        return (np.sign(x['badprob'].diff()).sum() == (x.shape[0]-1))or(np.sign(x['badprob'].diff()).sum() == -(x.shape[0]-1))
    
    @staticmethod
    def filter_fea(bin_dic,iv_threshold=0.2,mono=True):
        """
        按照IV值剔除相关性较高的特征
        """
        fea_info_list = []
        for k,v in bin_dic.items():
            mono_status = FeaEdaFunc.__check_monotony(v)
            iv = v['total_iv'][0]
            fea_info_list.append([k,mono_status,iv])
        fea_info = pd.DataFrame.from_records(fea_info_list,columns=['variable','mono_status','iv'])
        fea_info = fea_info.set_index('variable')
        return fea_info
    
    @staticmethod
    def calc_psi(bindic,exp,act):
        """
        根据分箱后的结果计算PSI
        test_bin_dicts 为woe后的结果
        """
        psi_dic = {}
        con_dic = {}
        for k,b in bindic.items():
            g1 = exp[[k]].fillna('missing')
            g2 = act[[k]].fillna('missing')
            cut_bin = b.loc[b['breaks']!= 'missing','breaks'].astype(float)
            cut_bin.loc[len(cut_bin)+1] = '-inf'
            cut_bin = cut_bin.astype(float).sort_values()
            g1['c'] = pd.cut(g1.loc[g1[k] != 'missing',k],bins=cut_bin,right=False)
            g2['c'] = pd.cut(g2.loc[g2[k] != 'missing',k],bins=cut_bin,right=False)
            g1['c'] = g1['c'].astype(str)
            g1.loc[g1[k] == 'missing','c'] = 'missing'
            count_a = g1.groupby('c').size()/len(g1)
            g2['c'] = g2['c'].astype(str)
            g2.loc[g2[k] == 'missing','c'] = 'missing'
            count_e = g2.groupby('c').size()/len(g2)
            con = pd.concat([count_a,count_e],axis=1,keys=['a','e'])
            con['a-e'] = con['a'] - con['e']
            con['ln(a/e)'] = np.log(con['a']/con['e'])
            con_dic[k] = con
            x = (con['a'] - con['e'])*np.log(con['a']/con['e'])
            x = x[~x.isnull()]
            psi = sum(x)
            psi_dic[k] = psi
        return psi_dic,con_dic

    @staticmethod
    def bin_save(bin_info,filename='bin_info.xlsx'):
        """
        保存分箱结果
        """
        with pd.ExcelWriter(filename) as writer:
            for n,d in bin_info.items():
                d.to_excel(writer, sheet_name=n)
    @staticmethod
    #逐步回归
    def __score_test(x_data,y_data,y_pred,DF=1):
        x_arr=np.matrix(x_data)
        y_arr=np.matrix(y_data).reshape(-1,1)
        yh_arr=np.matrix(y_pred).reshape(-1,1)
        grad_0=-x_arr.T * (y_arr-yh_arr)
        info_0=np.multiply(x_arr,np.multiply(yh_arr,(1-yh_arr))).T * x_arr
        cov_m=info_0**(-1)
        chi2_0=grad_0.T * cov_m * grad_0
        Pvalue=(1-scipy.stats.chi2.cdf(chi2_0[0,0],DF))
        return(chi2_0[0,0],DF,Pvalue)
    
    @staticmethod
    def step_logit(x_in, y_in, selection='stepwise',sle=0.1,sls=0.1,includes=[]):
        ###检查x和y的长度
        if len(x_in) != len(y_in):
            raise paraException(mesg='x,y do not have same length!')
        x_data = x_in.copy()
        y_data = y_in.copy()
        if isinstance(x_data,pd.core.frame.DataFrame):
            x_list = list(x_data.columns.copy())
        elif isinstance(x_data,np.ndarray):
            if len(x_data.shape) == 1:
                x_list = ['x_0']
                x_data = pd.DataFrame(x_data.reshape(-1,1),columns=x_list)
            elif len(x_data.shape) == 2:
                x_list = ['x_' + str(i) for i in np.arange(x_data.shape[1])]
                x_data = pd.DataFrame(x_data,columns=x_list)
            else:
                raise paraException(mesg='x error!')
        else:
            raise paraException(mesg='x error!')
        # 处理强制进入变量
        try:
            if (includes>0) and (includes>0) :
                includes = x_list[:includes].copy()
            else:
                includes = []
        except:
            pass
        # 处理x,y
        x_data['_const']=1
        if (isinstance(y_data,pd.core.frame.DataFrame) == True) or (isinstance(y_data,pd.core.series.Series) == True):
            y_data = y_data.values.reshape(-1,1)
        else:
            y_data = y_data.reshape(-1,1)
        # stepwise
        if selection.upper() == 'STEPWISE':
            include_list = ['_const'] + includes
            current_list = []
            candidate_list = [_x for _x in x_list if _x not in include_list]
            lgt = sm.Logit(y_data,x_data[include_list])
            res = lgt.fit()
            y_pred = res.predict()
            STOP_FLAG = 0 
            step_i = 1
            while(STOP_FLAG == 0):
                if len(candidate_list) == 0:
                    break
                score_list = [FeaEdaFunc.__score_test(x_data[include_list + current_list +[x0]]
                                        ,y_data
                                        ,y_pred)
                                for x0 in candidate_list]
                score_df = pd.DataFrame(score_list,columns=['chi2','df','p-vlue'])
                score_df['xvar'] = candidate_list
                slt_idx = score_df['chi2'].idxmax()
                p_value = score_df['p-vlue'].iloc[slt_idx]
                enter_x = candidate_list[slt_idx]
                if p_value <= sle:
                    current_list.append(enter_x)   ##加入模型列表
                    candidate_list.remove(enter_x) ##从候选变量列表平中删除
                else:
                    STOP_FLAG = 1
                lgt = sm.Logit(y_data,x_data[include_list+current_list])
                res = lgt.fit()
                y_pred = res.predict()
                chi2_df = res.wald_test_terms().table.copy()
                tmp_del_list = [tmp_x for tmp_x in chi2_df.index if tmp_x not in include_list]
                if len(tmp_del_list) > 0:
                    tmp_chi2 = chi2_df.loc[tmp_del_list].sort_values(by='statistic')
                    if tmp_chi2['pvalue'].iloc[0] >sls:
                        del_x = tmp_chi2.index[0]
                        if del_x == current_list[-1]:
                            current_list.remove(del_x)
                            STOP_FLAG = 1
                        else:
                            current_list.remove(del_x)
                        candidate_list.append(del_x)
                        lgt = sm.Logit(y_data,x_data[include_list+current_list])
                        res = lgt.fit()
                        y_pred = res.predict()
                    else:
                        pass
                else:
                    pass
                step_i += 1
            print(res.summary2())
            print(res.wald_test_terms())
            all_cols = list(x_in)
            all_params = list(res.params.index)
            remove_cols = [f for f in all_cols if f not in all_params]
            return remove_cols,res
        ####简单逻辑回归
        else:
            lgt = sm.Logit(y_data,x_data)
            res = lgt.fit()
            print(res.summary2())
            print(res.wald_test_terms())
            return res