import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import dill

'''归一化函数：将数据缩放到 [-1, 1]'''
def normalize_column(df):
    return 2 * (df - df.min()) / (df.max() - df.min()) - 1

# 自定义 pickle 模块包装器，用于处理 persistent id
class CustomDill:
    Unpickler = dill.Unpickler  # 提供 Unpickler 属性
    loads = staticmethod(dill.loads)
    
    @staticmethod
    def load(f, **kwargs):
        unpickler = dill.Unpickler(f, **kwargs)
        # 当遇到 persistent id 时直接返回 None
        unpickler.persistent_load = lambda pid: None
        return unpickler.load()

# 加载模型（结构与训练时保持一致）
def load_model(model_path, input_size, output_size, device):
    class Net(torch.nn.Module):
        def __init__(self, input_size, hidden_sizes=[2048, 1024, 512, 256], dropout_p=0.3, output_size=595):
            super(Net, self).__init__()
            layers = []
            in_size = input_size
            for h_size in hidden_sizes:
                layers.append(torch.nn.Linear(in_size, h_size))
                layers.append(torch.nn.BatchNorm1d(h_size))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout_p))
                in_size = h_size
            layers.append(torch.nn.Linear(in_size, output_size))
            self.layers = torch.nn.Sequential(*layers)
        
        def forward(self, x):
            x = self.layers(x)
            x = torch.nn.functional.softplus(x)
            return x

    model = Net(input_size=input_size, output_size=output_size)
    model.to(device)
    # 使用自定义 pickle 模块加载模型
    model.load_state_dict(torch.load(model_path, map_location=device, pickle_module=CustomDill))
    model.eval()
    return model

# 数据预处理函数
def preprocess_data(input_data):
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    input_data = pd.DataFrame(input_data).apply(normalize_column, axis=0)
    input_data = input_data.values.reshape(-1, 2683)
    return torch.tensor(input_data, dtype=torch.float32)

# 推理函数：使用模型进行前向传播，返回预测结果
def infer(model, input_data, device):
    input_tensor = preprocess_data(input_data).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return output.cpu().numpy()

# herb_filter 函数：加载模型并进行预测
def herb_filter(inputdata):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r"D:\myWork\github\nectar\data\weighted_herb_model_re_50.pth"
    
    model = load_model(model_path, input_size=2683, output_size=595, device=device)
    predictions = infer(model, inputdata, device)
    return predictions

# 示例调用
if __name__ == '__main__':
    sample_data = np.random.rand(2683)  # 单个样本
    preds = herb_filter(sample_data)
    print("预测结果：", preds)
