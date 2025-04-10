import csv
import gc
import os

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from model import RGCN
from utlis_con import graph_make, graph_contrastive_loss
import pandas as pd


print(torch.__version__)

gc.collect()
torch.cuda.empty_cache()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 'cuda' 这里如果没有指定具体的卡号,系统默认cuda:0

data, node_features,node_id,node_type,node_id_map, edge_index, edge_type, in_features_dim = graph_make('./data/grouped_data_filtered0121_福建.csv')
node_features= node_features.to(device)
node_id = node_id.to(device)
node_type = node_type.to(device)
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)


num_nodes = len(node_id)
num_relations = 26

###########################################

# 创建模型

hid_feature_dim = 128  # 隐藏层维度
out_feature_dim = 64
model = RGCN(in_features_dim, hid_feature_dim, out_feature_dim, num_relations)
model= model.to(device)


# print(model.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建学习率调度器，每10个epoch将学习率减少为原来的一半
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
max_grad_norm = 2.0

data = Data(x=node_features,
            node_id = node_id,
            edge_index= edge_index,
            edge_type = edge_type,
            num_nodes = num_nodes)

loader = NeighborLoader(data,
                      num_neighbors = [8,6],
                      batch_size = 512,
                      input_nodes = None)
# 进行训练
num_epochs =40  # 假设训练100轮


best_loss = float('inf')  # 将最小损失初始化为正无穷
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 前向传播
        # 前向传播，获取节点嵌入
        output = model(batch.x, batch.edge_index, batch.edge_type).to(device)

        # 计算图对比损失
        loss = graph_contrastive_loss(output, batch.edge_index)

        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()  # 更新模型参数

        total_loss += loss.item()


    scheduler.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.8f}')

    if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), './epoch/best_model1.pth')  # 保存模型参数
            print(f'Best model saved with loss: {best_loss:.8f}')

# # 收集嵌入和节点索引
# # 初始化一个张量，用于存储每个节点的嵌入
# total_output = torch.zeros((num_nodes, out_feature_dim), device='cuda')  # 全图嵌入
#
# # 存储每个节点是否已被访问（用于去重）
# visited_nodes = torch.zeros(num_nodes, dtype=torch.bool)
#
# model.eval()
# with torch.no_grad():
#     for batch_idx, batch in enumerate(loader):
#         batch = batch.to(device)
#
#         # 获取批次的输出和节点索引
#         batch_output = model(batch.x, batch.edge_index, batch.edge_type).cuda()
#         batch_input_id = batch.node_id.cuda()  # 替换为实际字段名称
#
#         # 更新总的嵌入张量
#         total_output[batch_input_id] = batch_output
#         visited_nodes[batch_input_id] = True
#
#         eval_batch_info.append({
#             "batch_idx": batch_idx,
#             "visited_node_count": batch_input_id.size(0)
#         })
#
# # 检查是否所有节点都已访问
# assert visited_nodes.all(), "Not all nodes were processed in the batches."
# print(f"Final total_output shape: {total_output.shape}")
# total_output_cpu = total_output.cpu()
# numpy_array0 = total_output_cpu.detach().numpy()
# df0 = pd.DataFrame(numpy_array0)
# with open('./embedding/Fujian_embedding1_64con.csv',mode='w',newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(numpy_array0)

