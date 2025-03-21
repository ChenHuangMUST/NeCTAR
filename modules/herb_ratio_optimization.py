import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr

def initialize_device(input_array):
    """初始化设备，选择CUDA或CPU"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.from_numpy(input_array).to(device)
    return input_tensor, device

def initialize_weights(herb_count, device):
    """初始化权重张量，并返回优化器"""
    weights = torch.rand(herb_count, device=device) * 0.1  # 初始化较小的正值  # 创建一个随机的权重张量
    weights.requires_grad_()  # 确保权重是叶子节点
    optimizer = torch.optim.Adam([weights], lr=0.0001)  # 降低学习率
    return weights, optimizer

def pearson_correlation(x, y):
    """计算皮尔逊相关系数"""
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    covariance = torch.mean((x - mean_x) * (y - mean_y))
    std_x = torch.std(x, unbiased=False)
    std_y = torch.std(y, unbiased=False)
    return covariance / (std_x * std_y + 1e-8)

def loss_function(input_tensor, weights):
    """定义新的损失函数，基于加权和与test的皮尔逊相关系数（负相关）"""
    # 确保所有的负权重都被设为0
    weights = torch.clamp(weights, min=0.0)
    # 将权重前添加一个 1.0
    #weights_new = torch.cat([torch.tensor([1.0], device=weights.device), weights])

    # 调整input_tensor并按行求和
    adjusted = input_tensor[:,1:] * weights
    sum_adjusted = adjusted.sum(dim=1)

    # 获取test（第一列）
    test = input_tensor[:, 0]

    # 计算皮尔逊相关系数
    pearson_corr = pearson_correlation(sum_adjusted, test)

    # 返回损失值和皮尔逊相关系数
    return 1 + pearson_corr, pearson_corr

def train_model(input_tensor, weights, optimizer, num_epochs=100000, tolerance=1e-4, patience=10):
    """训练模型并记录损失值"""
    loss_values = []
    no_improvement_count = 0  # 记录没有改进的周期数

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 应用 ReLU 激活函数，保证权重大于 0
        weights.data = torch.relu(weights.data)

        # 将权重限制在 0.0 到 10.0 之间
        with torch.no_grad():
            weights.data = torch.clamp(weights.data.to(input_tensor.device), min=0.0, max=10.0)

        # 计算损失和皮尔逊相关系数
        loss, pearson_corr = loss_function(input_tensor, weights)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 记录损失值
        loss_values.append(loss.item())

        # 打印调试信息
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Pearson Corr: {pearson_corr.item()}, Weights Mean: {weights.mean().item()}, Weights Std: {weights.std().item()}')

        # 监控损失变化情况，提前停止
        if epoch > 0 and abs(loss_values[-1] - loss_values[-2]) < tolerance:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if no_improvement_count >= patience:
            print(f'Early stopping at epoch {epoch}, Loss: {loss.item()}')
            break

    return weights, loss_values, epoch

def save_results(result_folder, weights, loss_values, epoch):
    """保存权重和损失图像"""
    # 根据当前时间生成一个时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # 在 weights 目录下创建以时间戳为名称的子目录
    weights_dir = os.path.join(result_folder, 'weights', timestamp)
    os.makedirs(weights_dir, exist_ok=True)

    # 保存权重到新建的时间戳子目录中
    weights_path = os.path.join(weights_dir, f'optimized_weights_epoch{epoch}.pth')
    torch.save(weights, weights_path)
    print(f"Saved weights to {weights_path}")
    
    # 保存损失曲线图（不调用 show()，只保存到 plots 目录）
    plots_dir = os.path.join(result_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - Epoch {epoch}')
    
    plot_path = os.path.join(plots_dir, f'loss_curve_epoch{epoch}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss curve to {plot_path}")

def optimize_weights(input_array, herb_count, result_folder):
    """主函数：执行初始化、训练、保存结果等流程"""
    input_tensor, device = initialize_device(input_array)
    weights, optimizer = initialize_weights(herb_count, device)
    weights, loss_values, epoch = train_model(input_tensor, weights, optimizer)
    save_results(result_folder, weights, loss_values, epoch)
    print(f'Optimized weights: {weights}')
    
    # 确保返回权重和损失值
    return weights, loss_values