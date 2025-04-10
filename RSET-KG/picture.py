import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from adjustText import adjust_text
#绘制关系相关##########################################################################################
plt.rc('font', family='Times New Roman')
# Set font to support Chinese characters
# plt.rcParams['font.sans-serif'] = ['SimHei']  # For Simplified Chinese
# plt.rcParams['axes.unicode_minus'] = False
# 读取CSV文件，假设第一列和第一行为行名和列名
df = pd.read_csv('F:/新型区域活动要素知识图谱/Figure/Fig4/关系图/指标关系表格.csv', index_col=0,encoding='utf-8')

# 创建一个空的无向图
G = nx.Graph()

# 遍历DataFrame并为每一对节点添加边
for row in df.index:
    for col in df.columns:
        weight = df.loc[row, col]
        if weight != 0:  # 可以根据实际需要设置阈值
            G.add_edge(row, col, weight=weight)

# 绘制图形
plt.figure(dpi=300,figsize=(5.2, 5.2))

# 使用circular_layout来让节点呈环形排列
pos = nx.circular_layout(G)

# 计算边的权重（用于设置边宽）
edges = G.edges(data=True)
weights = [data['weight'] for _, _, data in edges]

# 计算节点度数，以调整节点大小
node_size = [300 * np.sqrt(G.degree(node)) for node in G.nodes()]

# 计算边的透明度（用于边的可视化）
edge_alpha = np.array(weights) / max(weights)  # 边透明度按权重归一化

# 选择节点颜色和边的颜色
node_color = '#DCC7B6'  # 可以根据需求选择其他颜色
edge_color = plt.cm.viridis(np.array(weights) / max(weights))  # 采用渐变色显示边的权重

for i in range(0,len(edge_alpha)):
    if edge_alpha[i] == 1:
        edge_color[i, :] = [98 / 256, 143 / 256, 174 / 256, 1]
    elif edge_alpha[i] == 0.5:
        edge_color[i, :] = [210 / 256, 82 / 256, 73 / 256, 1]
    elif edge_alpha[i] == 0.6:
        edge_color[i, :] = [64 / 256, 192 / 256, 140 / 256, 1]


# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.7)

# 绘制边，边宽根据权重，颜色渐变
nx.draw_networkx_edges(G, pos, width=2, alpha=edge_alpha, edge_color=edge_color, style='solid')

# 绘制标签
# 这里我们先存储标签的位置和文本，然后使用 adjust_text 来调整标签位置
labels = {node: node for node in G.nodes()}
label_positions = {node: (pos[node][0], pos[node][1]) for node in G.nodes()}

# 调整标签位置以避免重叠
texts = [plt.text(label_positions[node][0], label_positions[node][1], node, fontsize=10, ha='center', va='center') for node in labels]
adjust_text(texts, only_move={'points': 'xy', 'text': 'xy'}, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

# 设置标题
#plt.title("Network Connectivity Graph (Circular Layout)", fontsize=16, fontweight='bold')

# 去掉坐标轴
plt.axis('off')
plt.savefig('F:/新型区域活动要素知识图谱/Figure/Fig4/关系图/指标关系2.jpg', bbox_inches='tight')
# 显示图形
plt.show()
##############################################################################################################################
# 示例二维数组和标签
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 示例二维数组
# from utlis_con import graph_make
#
# data = np.genfromtxt('output.csv', delimiter=',', skip_header=1)  # skip_header=1 用于跳过标题行
#
#
# # 每行数据的类型标签
#
# node_features,node_id,node_type,node_id_map, edge_index, edge_type, num_node_features = graph_make('./data/grouped_data_filtered0121.csv')
# label = node_type.cpu().numpy()
#
# # 创建标签到颜色的映射
#
#
# # 为每行根据标签赋予颜色
# # 创建一个与data相同形状的颜色矩阵，值由labels决定
# # 提取 x 和 y 坐标
# x = data[:, 0]
# y = data[:, 1]
#
#
# x = x[90110:97040]
# y = y[90110:97040,]
# label = label[90110:97040,]
# # 绘制散点图，颜色由 labels 决定
# plt.scatter(x, y, c=label, cmap='viridis')  # 使用'viridis'色图
# plt.colorbar()  # 显示颜色条，展示标签到颜色的映射关系
#
# # 添加标题和标签
# plt.title('Scatter Plot with Labels as Colors')
# plt.xlabel('X')
# plt.ylabel('Y')
#
# # 显示图形
# plt.show()