# data_io.py
import pandas as pd
import pickle

def load_herb_info(file_path):
    """加载中药信息"""
    return pd.read_excel(file_path)

def load_herb_nes(file_path):
    """加载中药 NES 数据"""
    return pd.read_csv(file_path, sep='\t')

def load_dosage_info(file_path):
    """加载中药的计量范围"""
    return pd.read_csv(file_path, sep='\t')
'''
def load_disease_data(file_path):
    """加载疾病 NES 数据"""
    with open(file_path, 'rb') as f:
        resultList = pickle.load(f)
    return pd.DataFrame(resultList)[['ID', "NES"]]
'''
def load_disease_data(file_path):
    """加载疾病 NES 数据"""
    # 假设txt文件是以制表符分隔的表格数据
    try:
        df = pd.read_csv(file_path, sep='\t')  # 如果是以逗号分隔，则使用 sep=','
    except:
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
    # 假设数据包含 'ID' 和 'NES' 两列
    return df[['ID', 'NES']]
