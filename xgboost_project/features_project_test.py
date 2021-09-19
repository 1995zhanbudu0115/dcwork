# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:46:59 2020

@author: zhanb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:30:12 2020

@author: zhanb
@project:特征整合与特征衍生
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


test_path = './data/test'
mode = 'test'

# 辅助画图
def plot_dist(ds):
    f,[ax1,ax2] = plt.subplots(1,2,figsize=(12,3))
    sns.boxplot(y=ds,ax=ax1)
    sns.distplot(ds,ax=ax2)
    
# 查看数据是否有缺失值    
def check_null_outlier(df):
    for c in list(df):
        df = df[df[c] != r'\N']
    return df

def fetch_trd_features():
    #交易数据
    trd_data = pd.read_csv(test_path+'/{}_trd.csv'.format(mode))
    #trd_features = trd_data[['id','flag']].drop_duplicates(subset=['id'])
    # 交易特征衍生
    # 近60天收支分类数据
    trd_amount = trd_data.groupby(['id','Dat_Flg1_Cd'])['cny_trx_amt'].sum()
    # 近60天收入总额
    trd_amount_in = trd_amount.loc[:,'C']
    # 近60天支出总额
    trd_amount_out = trd_amount.loc[:,'B']
    # 近60天收支总额
    trd_amount_all = trd_amount.groupby('id').sum()
    # 近60天收支记录次数数据
    trd_times = trd_data.groupby(['id','Dat_Flg1_Cd'])['cny_trx_amt'].count()
    # 近60天收入记录次数
    trd_times_in = trd_times.loc[:,'C']
    # 近60天支出记录次数
    trd_times_out = trd_times.loc[:,'B']
    # 近60天支出记录总次数
    trd_times_all = trd_times.groupby('id').sum()
    # 近60天收入最大值
    trd_amount_max = trd_data.groupby('id')['cny_trx_amt'].max().apply(
            lambda x: x if x > 0 else 0)
    # 近60天支出最大值
    trd_amount_min = trd_data.groupby('id')['cny_trx_amt'].min().apply(
            lambda x: x if x < 0 else 0)
    # 一级分类收支次数
    trd_times_categ1 = trd_data.groupby(['id','Trx_Cod1_Cd'])['cny_trx_amt'].count()
    # 多重索引数据unstack()
    trd_times_categ1 = trd_times_categ1.unstack().fillna(0)
    trd_times_categ1.columns=['times_1','times_2','times_3']
    # 二级分类收支次数
    trd_times_categ2 = trd_data.groupby(['id','Trx_Cod2_Cd'])['cny_trx_amt'].count()
    # 多重索引数据unstack()
    trd_times_categ2 = trd_times_categ2.unstack().fillna(0)
    # 给列名加前缀
    trd_times_categ2 = trd_times_categ2.add_prefix('times_')
    # 一级分类收支总额
    trd_amount_categ1 = trd_data.groupby(['id','Trx_Cod1_Cd'])['cny_trx_amt'].sum()
    trd_amount_categ1 = trd_amount_categ1.unstack().fillna(0)
    trd_amount_categ1.columns=['amount_1','amount_2','amount_3']
    # 二级级分类收支总额
    trd_amount_categ2 = trd_data.groupby(['id','Trx_Cod2_Cd'])['cny_trx_amt'].sum()
    trd_amount_categ2 = trd_amount_categ2.unstack().fillna(0)
    # 给列名加前缀
    trd_amount_categ2 = trd_amount_categ2.add_prefix('amount_')
    # 不同支付方式的交易次数
    trd_times_trdtype = trd_data.groupby(['id','Dat_Flg1_Cd','Dat_Flg3_Cd'])['cny_trx_amt'].count()
    trd_times_trdtype = trd_times_trdtype.unstack().unstack().fillna(0)
    trd_times_trdtype.columns = ['type_A_B','type_A_C','type_B_B','type_B_C','type_C_B','type_C_C']
    # 近60天收支均值
    trd_amount_mean = trd_data.groupby(['id','Dat_Flg1_Cd'])['cny_trx_amt'].mean()
    # 近60天收入均值
    trd_amount_mean_in = trd_amount_mean.loc[:,'C']
    # 近60天支出均值
    trd_amount_mean_out = trd_amount_mean.loc[:,'B']
    
    ser_list = [trd_amount_in,trd_amount_out,trd_amount_all,trd_times_in,
                trd_times_out,trd_times_all,trd_amount_max,trd_amount_min,
                trd_times_categ1,trd_times_categ2,trd_amount_categ1,
                trd_amount_categ2,trd_times_trdtype,trd_amount_mean_in,trd_amount_mean_out]
    trd_cols = ['id','trd_amount_in','trd_amount_out','trd_amount_all',
                'trd_times_in','trd_times_out','trd_times_all',
                'trd_amount_max','trd_amount_min']
    trd_features = pd.concat(ser_list,axis=1).reset_index().fillna(0)
    trd_cols.extend(list(trd_features)[9:])
    trd_features.columns=trd_cols
    trd_features.to_csv(test_path+'/trd_features.csv',index=False)
    return trd_features
    
def fetch_beh_features():
    #用户app行为数据
    beh_data = pd.read_csv(test_path+'/{}_beh.csv'.format(mode))
    beh_data['date'] = beh_data['page_tm'].str.split(' ').str.get(0)
    # 近30天每个用户访问页面的总次数
    beh_times_page = beh_data.groupby(['id','page_no'])['page_tm'].count().unstack().fillna(0)
    beh_times_page = beh_times_page.add_prefix('beh_times_page_')
    ben_times_app = beh_data.groupby(['id'])['page_tm'].count()
    # 近30天每个用户访问app的日均次数
    beh_times_daily = beh_data.groupby(['id','date'])['page_tm'].count().groupby('id').mean()
    ser_list = [beh_times_page,ben_times_app,beh_times_daily]
    beh_features = pd.concat(ser_list,axis=1).reset_index()
    beh_features.rename(columns={'index':'id'},inplace=True)
    beh_features.to_csv(test_path+'/beh_features.csv',index=False)
    
def fetch_tag_features():
    #用户标签数据
    #tag_data = pd.read_csv('./data/train/train_tag.csv')
    tag_data = pd.read_csv(test_path+'/{}_tag.csv'.format(mode))
#    查看性别数据，发现有缺失，但缺失数据比例小于百分之2，
#    所以性别缺失数据可以剔除
    tag_data.loc[tag_data['gdr_cd'] == '\\N','gdr_cd'] = 0
    tag_data.loc[tag_data['gdr_cd']=='F','gdr_cd'] = 0
    tag_data.loc[tag_data['gdr_cd']=='M','gdr_cd'] = 1
    # 查看箱型图与分布图
    plot_dist(tag_data['age'])
    # 查看年龄最大值与最小值，最大值84.最小值19，在合理范围内
    tag_data['age'].min()
    tag_data['age'].max()
    # 查看婚姻状况，未填写婚姻状况的比例小于百分之1，将缺失数据剔除，并进行onehot编码
    tag_data.loc[tag_data['mrg_situ_cd'] == '~','mrg_situ_cd'] = 'B'
    tag_data.loc[tag_data['mrg_situ_cd'] == '\\N','mrg_situ_cd'] = 'B'
    tag_mrg = pd.get_dummies(tag_data['mrg_situ_cd']).add_prefix('mrg_')
    # 学位
    tag_data.loc[tag_data['deg_cd']=='~','deg_cd']='E'
    tag_data.loc[tag_data['deg_cd'].isna(),'deg_cd']='F'
    tag_deg = pd.get_dummies(tag_data['deg_cd']).add_prefix('deg_')
    # 教育程度
    tag_data.loc[tag_data['edu_deg_cd']=='~','edu_deg_cd']='E'
    tag_data.loc[tag_data['edu_deg_cd'].isnull(),'edu_deg_cd'] = 'L'
    tag_data.loc[tag_data['edu_deg_cd']=='\\N','edu_deg_cd']='E'
    tag_edu_deg = pd.get_dummies(tag_data['edu_deg_cd']).add_prefix('edu_deg_')
    # 学历
    tag_data = tag_data[~tag_data['acdm_deg_cd'].isnull()]
    tag_acdm_deg = pd.get_dummies(tag_data['acdm_deg_cd']).add_prefix('acdm_deg_')
    # 转账类型
    tag_data = tag_data[tag_data['atdd_type'] != '\\N']
    tag_data['atdd_type'] = tag_data['atdd_type'].astype(float)
    tag_data.loc[tag_data['atdd_type'].isnull(),'atdd_type'] = 0
    # 工作年限
    tag_data.loc[tag_data['job_year']=='\\N','job_year'] = 0
    tag_data['job_year'] = tag_data['job_year'].astype(float)
    tag_data = tag_data[tag_data['job_year']<50]
    # 
    tag_data = tag_data.fillna(0)
    ser_list = [['mrg_situ_cd',tag_mrg],['deg_cd',tag_deg],
                  ['edu_deg_cd',tag_edu_deg],['acdm_deg_cd',tag_acdm_deg]]
    drop_index = [c[0] for c in ser_list]
    ser_list = [c[1] for c in ser_list]
    tag_features = pd.concat(ser_list,axis=1)
    tag_data.drop(drop_index,axis=1,inplace=True)
    tag_data = pd.concat([tag_data,tag_features],axis=1)
    tag_data.to_csv(test_path+'/tag_features.csv',index=False)
    return tag_data
    
def combine_features():
    if os.path.exists(test_path+'/beh_features.csv'):
        beh_features = pd.read_csv(test_path+'/beh_features.csv')
    else:
        beh_features = fetch_beh_features()
        
    if os.path.exists(test_path+'/trd_features.csv'):
        trd_features = pd.read_csv(test_path+'/trd_features.csv')
    else:
        trd_features = fetch_trd_features()
        
    if os.path.exists(test_path+'/tag_features.csv'):
        tag_features = pd.read_csv(test_path+'/tag_features.csv')
    else:
        tag_features = fetch_tag_features()
    features = tag_features.merge(trd_features,on='id', how='left')
    features = features.merge(beh_features,on='id',how='left')
    features.replace('\\N',0,inplace=True)
    features = features.fillna(0)
    features.to_csv(test_path+'/features.csv',index=False)
    return features
    
def select_features():
    if os.path.exists(test_path+'/features.csv'):
        features = pd.read_csv(test_path+'/features.csv')
    else:
        features = combine_features()
        
if __name__ == '__main__':
    combine_features()