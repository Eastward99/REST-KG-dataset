import csv

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from model import RGCN
from utlis_con import graph_make

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, node_features,node_id,node_type,node_id_map, edge_index, edge_type, in_features_dim = graph_make('./data/grouped_data_filtered0121.csv')
node_features= node_features.to(device)
node_id = node_id.to(device)
node_type = node_type.to(device)
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)
num_nodes = len(node_id)


# 模型参数
in_features_dim = 76  # 输入特征维度
hid_feature_dim = 64  # 隐藏特征维度
out_feature_dim = 32   # 输出特征维度
num_relations = 26     # 关系数量

data_id = data.iloc[:, -26:]
# 实例化模型
model_test = RGCN(in_features_dim, hid_feature_dim, out_feature_dim, num_relations)

# 加载保存的模型权重
model_file = './epoch/best_model100.pth'  # 修改为你的文件路径
model_test.load_state_dict(torch.load(model_file))


model_test = model_test.to(device)

# 切换到评估模式
model_test.eval()

print("模型已成功加载！")

with torch.no_grad():
    all_embeddings = model_test(node_features, edge_index, edge_type)

embeddings = all_embeddings.cpu().detach().numpy()
print("embedding已成功加载！")
with open('./embedding/China_embedding100_32con.csv',mode='w',newline='') as file:
    writer = csv.writer(file)
    writer.writerows(embeddings)

similarity_dict = {}
for time, row in data_id.iterrows():
    gdp_id = row['GDP_id']  # 获取该时间点的GDP实体ID
    gdp_embedding = embeddings[gdp_id]   # 获取GDP实体的嵌入向量

    similarity_dict[time] = {}

    # 遍历每个实体（排除GDP实体）
    for entity, entity_id in row.items():
        if entity != 'GDP_id':  # 排除GDP实体本身
            entity_embedding = embeddings[entity_id]  # 获取当前实体的嵌入向量（假设实体ID从1开始）

            # 计算GDP实体与当前实体的相似度
            similarity = cosine_similarity([gdp_embedding], [entity_embedding])[0][0]

            # 将相似度存入字典中，键是实体名字，值是相似度
            similarity_dict[time][entity] = similarity

# 将字典转换为DataFrame
similarity_df = pd.DataFrame(similarity_dict)
similarity_df.to_csv("similarity_results100.csv")
print("GDP_id已成功加载！")

