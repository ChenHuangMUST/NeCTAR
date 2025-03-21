"""用量2权重"""
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
    
def dosage_to_weight(dosage_array, usageRange):
    usage_list = []
    for i in range(len(dosage_array)):
        usage = normalize_to_range([0,usageRange[i][1],dosage_array[i]], 0, 1)[-1]
        #if usage >= usageRange[i][0]:
        usage_list.append(round(usage, 1))
        #else:
        #    usage_list.append(0)
    return usage_list
