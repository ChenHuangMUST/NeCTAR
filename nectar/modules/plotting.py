# plotting_module.py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pickle

def plot_optimization_results(all_results, result_folder):
    """
    根据所有迭代结果绘制图像并保存到指定的文件夹中。
    
    参数:
        all_results: 一个列表，每个元素为一次迭代的结果字典，包含 'formula', 'normalized_weight_list' 和 'score' 等信息。
        result_folder: 存储结果文件的文件夹路径（通常为时间戳命名的文件夹）。
    """
    # 设置 matplotlib 配置（中文和负号支持）
    mpl.rcParams.update({
        'font.family': 'SimHei',          # 使用支持中文的字体
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'mathtext.default': 'regular',
    })
    
    # 如果 all_results 非空，则获取各迭代评分及最佳迭代索引（最小得分为最优）
    if all_results:
        scores_list = [result['score'] for result in all_results]
        best_score = min(scores_list)
        best_index = scores_list.index(best_score)
    else:
        best_index = 0
        best_score = None

    # 准备评分和每次迭代的中药权重信息
    scores = []
    herb_weights = []
    for result in all_results[:best_index+1]:
        scores.append(result['score'])
        herb_weight = dict(zip(result['formula'], result['normalized_weight_list']))
        herb_weights.append(herb_weight)
    
    # 构造一个固定顺序的中药列表
    herb_list = []
    seen = set()
    for result in all_results[:best_index+1]:
        for herb in result['formula']:
            if herb not in seen:
                seen.add(herb)
                herb_list.append(herb)
    
    # 创建画布，包含上部的中药权重分布图和下部的评分曲线图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), 
                                     gridspec_kw={'height_ratios': [2, 1]},
                                     sharex=True)
    
    # 上部：中药权重分布图，采用 viridis 配色方案
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('viridis')
    for iter_idx, current_weights in enumerate(herb_weights):
        for herb_idx, herb in enumerate(herb_list):
            if herb in current_weights:
                ax1.scatter(iter_idx, herb_idx, 
                            color=cmap(current_weights[herb]),
                            s=40,
                            edgecolor='w',
                            linewidth=0.5,
                            zorder=2)
    ax1.set_yticks(np.arange(len(herb_list)))
    ax1.set_yticklabels(herb_list)
    ax1.set_ylabel('Herbal Composition', labelpad=10)
    ax1.tick_params(axis='y', which='both', length=0)  # 隐藏刻度线
    ax1.grid(axis='x', linestyle='--', alpha=0.5, zorder=1)
    ax1.set_title('Evolution of Herbal Composition (Color Intensity Represents Relative Weight)')
    
    # 添加 colorbar
    cax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cax, label='Normalized Weight')
    
    # 下部：评分曲线图
    ax2.plot(scores, 
             color='#2C5F8A', 
             marker='o',
             markersize=8,
             markeredgecolor='w',
             linewidth=2,
             zorder=2)
    # 标注最佳点
    if best_score is not None:
        ax2.scatter(best_index, best_score,
                    color='#B22222',
                    marker='*',
                    s=200,
                    edgecolor='w',
                    linewidth=1,
                    zorder=3,
                    label=f'Optimal Value (Iteration {best_index+1})')
    # 添加每个点的数值标签
    for i, (x, y) in enumerate(zip(range(len(scores)), scores)):
        ax2.text(x, y, f'{y:.2f}',
                 ha='center', va='bottom',
                 fontsize=10,
                 color='#2C5F8A')
    
    ax2.set_xlabel('Iteration Number', labelpad=10)
    ax2.set_ylabel('Objective Function Value')
    ax2.set_xticks(np.arange(len(scores)))
    ax2.set_xticklabels([f'Iteration {i+1}' for i in range(len(scores))])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=False, loc='upper right')
    
    # 统一设置坐标轴样式
    for ax in [ax1, ax2]:
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    
    # 调整整体布局
    plt.subplots_adjust(left=0.15, right=0.9, hspace=0.08)
    
    # 保存图像到 result_folder（建议保存为 PDF 格式的矢量图）
    output_filename = os.path.join(result_folder, "optimization_plot.pdf")
    plt.savefig(output_filename, format="pdf", bbox_inches="tight", dpi=300)
    
    # 保存所有优化结果到 result_folder
    with open(os.path.join(result_folder, 'all_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
