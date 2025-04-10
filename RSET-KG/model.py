import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data


# 定义R-GCN模型
class RGCN(nn.Module):
    def __init__(self,num_node_features , hidden_dim, num_classes, num_relations):
        super(RGCN, self).__init__()

        # 第一层 R-GCN
        self.conv1 = RGCNConv(num_node_features, hidden_dim, num_relations)
        # 第二层 R-GCN
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations)

    def forward(self, x, edge_index, edge_type):
        # 第一层卷积
        x = F.relu(self.conv1(x, edge_index, edge_type))
        # 第二层卷积
        x = self.conv2(x, edge_index, edge_type)
        return x

# 定义模型
# class RGCN(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats, rel_num):
#         super(RGCN, self).__init__()
#         self.conv1 = RelGraphConv(in_feats, hid_feats, rel_num)
#         self.conv2 = RelGraphConv(hid_feats, out_feats, rel_num)
#
#     def forward(self, edge_index, edge_type, feats):
#         h = self.conv1((feats, feats), edge_index, edge_type)
#         h = torch.relu(h)
#         h = self.conv2((h, h), edge_index, edge_type)
#         return h


# edges_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[1, 2, 3, 0, 1, 2, 3, 4, 5, 6]],dtype=torch.long)
#
# rel_type = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
#
#
#
# # 创建模型
# in_feats = 3
# hid_feats = 4
# out_feats = 2
# rel_num = 4
# model = RGCN(in_feats, hid_feats, out_feats, rel_num)
#
# # 随机生成特征
# features = torch.randn((10, 3))
#
# # 计算输出
# output = model(features, edges_index, rel_type)
# print(output)