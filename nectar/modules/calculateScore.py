import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
# ==============================================
# 全局设置中文字体（根据操作系统选择）
# ==============================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==============================================
# 自定义函数：绘制柱状图（模仿甘草/人参示例）
# ==============================================


def plot_custom_bar(sum_pos, sum_neg, test_pos, test_neg, save_path):
    # 准备数据
    labels = ['sum_adjusted', 'test']
    pos_counts = [sum_pos, test_pos]
    neg_counts = [sum_neg, test_neg]

    # 绘图参数
    x = np.arange(len(labels))
    width = 0.35
    colors = {'positive': '#FF6F61', 'negative': '#20B2AA'}

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, pos_counts, width, label='positive', color=colors['positive'])
    bars2 = ax.bar(x + width/2, neg_counts, width, label='negative', color=colors['negative'])

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    # 设置图形属性
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Positive and Negative Value Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    
    # 保存为SVG格式
    plt.savefig(save_path, format='svg')
    plt.close()

def calculateScore(input_tensor, weights_cpu_list):
    # 确保所有的负权重都被设为0
    weights_cpu_list = [max(0, weight) for weight in weights_cpu_list]
    
    # 调整input_tensor并按行求和
    adjusted = input_tensor[:, 1:] * weights_cpu_list
    sum_adjusted = adjusted.sum(axis=1)

    # 获取test（第一列）
    test = input_tensor[:, 0]
    n = 1000
    top_list = [10*(1 - i/n) for i in range(n)] + [0]
    scorelist = []
    adjusted_list = []
    
    for top in top_list:
        # 计算斯皮尔曼相关系数
        spearman_corr, _ = spearmanr(sum_adjusted*top, test)
        # 保留原始求和逻辑（但可能需要检查业务逻辑合理性）
        
        scorelist.append(spearman_corr)
        adjusted_list.append(sum_adjusted)
    # 返回最优分数及对应的权重列表
    spearman_corr = min(scorelist)
    sum_adjusted = adjusted_list[scorelist.index(spearman_corr)]
    input_array = np.column_stack((test, sum_adjusted))
    input_tensor_summed = np.sum(input_array, axis=1, keepdims=True)
    return input_tensor_summed, spearman_corr




def calculateScore_plot(formula, input_tensor, weights_cpu_list, result_folder, plot=0):
    # 设置PDF字体类型为可编辑的TrueType字体
    mpl.rcParams['pdf.fonttype'] = 42
    
    # 确保所有的负权重都被设为0
    weights_cpu_list = [max(0, weight) for weight in weights_cpu_list]
    
    # 调整input_tensor并按行求和
    adjusted = input_tensor[:, 1:] * weights_cpu_list
    sum_adjusted = adjusted.sum(axis=1)

    # 获取test（第一列）
    test = input_tensor[:, 0]
    # 计算test值为0的蒙版
    test_mask = (test == 0)
    #sum_adjusted[test_mask] = 0
    # 删除test值为0的行
    test = test[~test_mask]
    # 删除sum_adjusted值为0的行
    sum_adjusted = sum_adjusted[~test_mask]
    # 计算斯皮尔曼相关系数
    spearman_corr, _ = spearmanr(sum_adjusted, test)
    
    # 标准化数据到[-1, 1]范围
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sum_adjusted_scaled = scaler.fit_transform(sum_adjusted.reshape(-1, 1)).flatten()
    test_scaled = scaler.fit_transform(test.reshape(-1, 1)).flatten()
    
    # 绘制相关性散点图
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    sns.regplot(x=sum_adjusted_scaled, y=test_scaled, scatter_kws={'alpha':0.5, 'color': 'dodgerblue'}, line_kws={'color': 'darkorange', 'linewidth': 2})
    plt.title(f'Spearman Correlation: {spearman_corr:.2f}', fontsize=16)
    plt.xlabel('formula score (Standardized)', fontsize=14)
    plt.ylabel('disease score (Standardized)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{result_folder}/spearman_scatter_{plot}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # 绘制全变量斯皮尔曼热力图
    corr_matrix, _ = spearmanr(input_tensor, axis=0)


    try:
        plt.figure(figsize=(12, 8))  # 增加图形的高度
        sns.set(font_scale=1.0)  # 调整字体大小
        
        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
            xticklabels=["disease"] + formula,  # 使用中文标签
            yticklabels=["disease"] + [f'herb{i+1}' for i in range(corr_matrix.shape[0]-1)],  # 使用中文标签
            vmin=-1, vmax=1,  # 确保颜色条的范围对称
            cbar_kws={'shrink': 0.8}
        )
        plt.title('Spearman Correlation Matrix', fontsize=16)  # 使用中文标题
        plt.xticks(rotation=45, ha='right')  # 确保 x 轴文本倾斜 45 度并右对齐
        plt.yticks(rotation=0)
        plt.tight_layout()  # 自动调整布局
        plt.savefig(f'{result_folder}/spearman_heatmap_{plot}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"绘制热力图时出错: {e}")  # 使用中文错误提示
    
        # 数据预处理
    weights_cpu_list = [max(0, w) for w in weights_cpu_list]
    adjusted = input_tensor[:, 1:] * weights_cpu_list
    sum_adjusted = adjusted.sum(axis=1)

    # 处理test列
    test = input_tensor[:, 0]
    
    # 保存 test 和 sum_adjusted 的值
    np.save(f"{result_folder}/test_values_{plot}_0.npy", test)
    np.save(f"{result_folder}/sum_adjusted_values_{plot}_0.npy", sum_adjusted)

    # ================= 绘制韦恩图 =================
    plt.figure(figsize=(8, 6))

    # 定义韦恩图的集合
    set_dis_pos = set(np.where(test > 0)[0])  # test 正数的索引
    set_formula_pos = set(np.where(sum_adjusted > 0)[0])  # sum_adjusted 正数的索引

    set_dis_neg = set(np.where(test < 0)[0])  # test 负数的索引
    set_formula_neg = set(np.where(sum_adjusted < 0)[0])  # sum_adjusted 负数的索引

    # 定义每个椭圆的属性
    ellipses = [
        {"center": (2, 3), "width": 5, "height": 2.2, "angle": -25, "label": "Dis_pos", "color": (128/255, 223/255, 255/255), "alpha": 0.7},
        {"center": (3, 3.5), "width": 5, "height": 2, "angle": -25, "label": "Dis_neg", "color": (190/255, 255/255, 129/255), "alpha": 0.7},
        {"center": (4.5, 3), "width": 5, "height": 2.2, "angle": 25, "label": "For_pos", "color": (128/255, 246/255, 246/255), "alpha": 0.7},
        {"center": (3.5, 3.5), "width": 5, "height": 2, "angle": 25, "label": "For_neg", "color": (255/255, 162/255, 127/255), "alpha": 0.7}
    ]

    ax = plt.gca()

    # 绘制椭圆的填充部分
    for ellipse in ellipses:
        facecolor = ellipse["color"] + (ellipse["alpha"],)
        ell = Ellipse(xy=ellipse["center"], width=ellipse["width"], height=ellipse["height"], angle=ellipse["angle"],
                      edgecolor='none', facecolor=facecolor, label=ellipse["label"])
        ax.add_patch(ell)

    # 绘制椭圆的边缘部分
    for ellipse in ellipses:
        ell = Ellipse(xy=ellipse["center"], width=ellipse["width"], height=ellipse["height"], angle=ellipse["angle"],
                      edgecolor='black', facecolor='none', linewidth=1, label='_nolegend_')
        ax.add_patch(ell)

    # 手动添加文本
    texts = [
        (3.25, 2, str(len(set_dis_pos&set_formula_pos))), (3.25, 4.05, str(len(set_dis_neg&set_formula_neg))),
        (2.6, 2.4, "0"), (3.9, 2.4, "0"),
        (3.25, 2.85, "0"), (3.25, 2.85, "0"),
        (1.8, 2.5, str(len(set_dis_pos&set_formula_neg))), (4.95, 2.5, str(len(set_dis_neg&set_formula_pos))),
        (2.25, 3.3, "0"), (4.25, 3.3, "0"),
        (1.6, 4, "0"), (5.15, 4, "0"),
        (0.5, 3.5, str(len(set_dis_pos)-len(set_dis_pos&set_formula_neg)-len(set_dis_pos&set_formula_pos))), (6.25, 3.5, str(len(set_dis_neg)-len(set_dis_neg&set_formula_pos)-len(set_dis_neg&set_formula_neg))),
        (1.9, 4.5, "0"), (4.85, 4.5, "0"),
    ]

    for x, y, text in texts:
        plt.text(x, y, text, fontsize=12, ha='center', va='center', color='black')

    # 设置标签和范围
    plt.xlim(-1, 7)
    plt.ylim(0, 7)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.title("Pathway", fontsize=16)
    plt.legend(loc='upper left')

    # 保存为矢量图
    plt.savefig(f"{result_folder}/venn_diagram_{plot}.svg", format="svg", bbox_inches="tight")
    plt.close()

    mask = (test != 0)
    test = test[mask]
    sum_adjusted = sum_adjusted[mask]

    # 计算统计量
    sum_pos = np.sum(sum_adjusted > 0)
    sum_neg = np.sum(sum_adjusted < 0)
    test_pos = np.sum(test > 0)
    test_neg = np.sum(test < 0)

    # ================= 绘制柱状图 =================
    plot_custom_bar(
        sum_pos, sum_neg, test_pos, test_neg,
        f"{result_folder}/barplot_{plot}.svg"
    )

    # 保存 test 和 sum_adjusted 的值
    np.save(f"{result_folder}/test_values_{plot}_1.npy", test)
    np.save(f"{result_folder}/sum_adjusted_values_{plot}_1.npy", sum_adjusted)


    # ================= 绘制热力图 =================
    def normalize_data(data):
        """将数据标准化到 [-1, 1] 范围"""
        return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

    def hotmap_plot(test_data, sum_adjusted_data, plot):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False    # 确保负号正常显示

        # 标准化数据
        test_data_normalized = normalize_data(test_data)
        sum_adjusted_data_normalized = normalize_data(sum_adjusted_data)

        # 选取前100大的绝对值位置
        top_100_indices = np.argsort(np.abs(test_data_normalized))[-500:][::-1]
        test_loaded_mini = test_data_normalized[top_100_indices]
        sum_adjusted_loaded_mini = sum_adjusted_data_normalized[top_100_indices]

        # 根据 test_loaded_mini 的值进行排序（考虑正负）
        sort_indices = np.argsort(test_loaded_mini)[::-1]  # 降序排序索引
        sorted_test_loaded_mini = test_loaded_mini[sort_indices]
        sorted_sum_adjusted_loaded_mini = sum_adjusted_loaded_mini[sort_indices]

        # 调整为竖直方向，按列堆叠
        combined_data = np.column_stack((sorted_test_loaded_mini, sorted_sum_adjusted_loaded_mini))

        # 使用 seaborn 绘制热图
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            combined_data,
            center=0,
            cmap='coolwarm',
            cbar_kws={'label': 'Value'},
            xticklabels=['Diease score', 'Formula score']  # 两列对应的标签
        )
        plt.title('Heatmap of Diease score and Formula score', fontsize=14)
        plt.xlabel('Data Type', fontsize=14)
        plt.ylabel('Rows (Sorted)', fontsize=14)
        plt.tight_layout()

        # 保存热图
        plt.savefig(f"{result_folder}/heatmap_{plot}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

    # 调用函数绘制热力图
    hotmap_plot(test, sum_adjusted, plot)
