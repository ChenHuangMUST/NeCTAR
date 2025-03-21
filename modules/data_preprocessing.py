'''数据准备'''
import pandas as pd
import numpy as np
import pickle

def keep_top_100_values(data):
    # 创建一个与输入数据形状相同的全零矩阵
    result = np.zeros_like(data)
    
    # 找到绝对值前200大的值的索引
    top_100_indices = np.argsort(np.abs(data))[-200:]
    
    # 将这些值保留，其他值设置为0
    result[top_100_indices] = data[top_100_indices]
    
    return result
def normalize_column(df):
    """标准化列数据到-1到1的范围"""
    try:
        return 2 * (df - df.min()) / (df.max() - df.min()) - 1
    except:
        print("Error:", df)
def prepare_herb_data(df_herb_nes, formula, nor = False):
    """处理中药数据，包括填充缺失值和标准化"""
    df_herbs = df_herb_nes[formula]
    df_herbs.fillna(0, inplace=True)  # 填充nan值为0
    if nor == True:
        df_herbs = df_herbs.apply(normalize_column, axis=0)  # 标准化数据
    return df_herbs

def merge_disease_data(df_herb_nes, df_disease):
    """将疾病数据与中药 NES 数据按ID合并"""
    result = pd.merge(df_herb_nes, df_disease, how='left', on='ID')
    result.fillna(0, inplace=True)  # 将缺失值填充为0
    print(result.shape)
    return result

'''
def process_disease_data(result):
    """处理疾病数据，提取NES并进行标准化"""
    result = pd.DataFrame(keep_top_100_values(result["NES"]))
    result = result.apply(normalize_column, axis=0)
    return result
'''
def process_disease_data(result, nor = False):
    """处理疾病数据，提取NES并进行标准化"""
    result = pd.DataFrame(result["NES"])
    if nor == True:
        result = result.apply(normalize_column, axis=0)
    return np.array(result["NES"])
def prepare_input_data(df_herb_nes, formula, df_disease):
    """准备最终的输入数据"""
    
    # 准备中药数据
    df_herbs = prepare_herb_data(df_herb_nes, formula, nor = True)

    # 合并疾病数据
    result = merge_disease_data(df_herb_nes, df_disease)

    # 处理疾病数据
    result_nes = process_disease_data(result, nor = True)
    #result_nes = keep_top_100_values(result_nes)

    # 合并疾病数据与方剂数据作为输入
    input_array1 = np.column_stack((result_nes, df_herbs))
    
    # 没有归一化的数据
    # 准备中药数据
    df_herbs = prepare_herb_data(df_herb_nes, formula)

    # 合并疾病数据
    result = merge_disease_data(df_herb_nes, df_disease)

    # 处理疾病数据
    result_nes = process_disease_data(result)
    #result_nes = keep_top_100_values(result_nes)

    # 合并疾病数据与方剂数据作为输入
    input_array2 = np.column_stack((result_nes, df_herbs))
    
    return input_array1,input_array2
