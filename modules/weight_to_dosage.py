"""权重2用量"""
def normalize_list(weights_list):
    if not weights_list:
        return []
    
    min_value = min(weights_list)
    max_value = max(weights_list)
    
    if min_value == max_value:
        # 如果所有值都相同，直接返回全0的列表
        return [0.0] * len(weights_list)
    
    normalized_list = [(value - min_value) / (max_value - min_value) for value in weights_list]
    return normalized_list

def normalize_to_range(data, a, b):
    # 计算原始数据的最小值和最大值
    min_val = min(data)
    max_val = max(data)
    
    # 如果数据的最小值和最大值相同，直接返回目标范围的均值
    if min_val == max_val:
        return [a + (b - a) / 2] * len(data)
    
    # 标准化数据
    normalized_data = [a + (x - min_val) * (b - a) / (max_val - min_val) for x in data]
    
    return normalized_data
    
def weightToDosage(weights_list, usageRange):
    # 将张量移动到 CPU
    weights_cpu = weights_list.cpu()
    # 将张量转换为列表
    weights_weights_list = normalize_list(weights_cpu.tolist())
    weights_list = weights_cpu.tolist()
    usage_list = []
    for i in range(len(weights_weights_list)):
        usage = normalize_to_range([0,1,weights_weights_list[i]], usageRange[i][0], usageRange[i][1])[-1]
        #if usage >= usageRange[i][0]:
        if usage - usageRange[i][0] < 0.01:
            usage = 0
        usage_list.append(round(usage, 1))
        #else:
        #    usage_list.append(0)
    return weights_list, usage_list, weights_weights_list

