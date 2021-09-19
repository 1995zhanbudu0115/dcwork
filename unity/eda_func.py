import pandas as pd
import numpy as np

class EdaFunc:
    
    @staticmethod
    def get_features_type(xfea):
        """
       获取特征类型及其对应的属性值数量
       """
        types_ser = xfea.dtypes
        cat_feas = types_ser[(types_ser == object)|(types_ser == str)].index
        cat_feas_dic = {c: len(xfea[c].unique()) for c in cat_feas}
        num_feas = types_ser[(types_ser==float)].index
        num_feas_dic = {n: len(xfea[n].unique()) for n in num_feas if len(xfea[n].unique()) >= 3}
        return {'cat_feas':pd.Series(cat_feas_dic),'num_feas':pd.Series(num_feas_dic)}
    
    @staticmethod
    def get_numfeas(df): 
        """
       获取数值变量
       """
        type_ser = df.dtypes
        numFeas = type_ser[(type_ser == 'float')|(type_ser == 'int')]
        numFeas = {c : len(df[c].unique()) for c in numFeas.index}
        numFeas = pd.Series(numFeas)
        return numFeas[numFeas>=6],numFeas[numFeas<6]

if __name__ == "__main__":
    EdaFunc.get_numfeas()
            