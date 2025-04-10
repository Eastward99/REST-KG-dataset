import pandas as pd

# 读取 CSV 文件，假设文件路径为 'your_file.csv'
df = pd.read_csv('F:/新型区域活动要素知识图谱/Figure/Fig_sim/china_similarity_results_city.csv')

# 假设第一列是年份，第二列是城市，第三列到最后一列是需要比较的数据
data_columns = df.iloc[:, 2:]

# 计算每行数据中的最大值
max_values = data_columns.max(axis=1)

# 获取每行最大值对应的列名
max_columns = data_columns.idxmax(axis=1)

# 创建一个新的 DataFrame 保存结果
result_df = pd.DataFrame({
    '年份': df['年份'],
    '城市': df['所属城市'],
    '最大值': max_values,
    '对应列名': max_columns
})

# 将结果保存为 CSV 文件
result_df.to_csv('F:/新型区域活动要素知识图谱/Figure/Fig_sim/城市最大值_中国100.csv', index=False, encoding='utf-8')  # 设置 index=False 来避免保存索引列

print("结果已保存为 'result.csv'")

# import pandas as pd
# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # 读取CSV文件
# data = pd.read_csv('F:/新型区域活动要素知识图谱/Figure/Fig4/知识嵌入/china_embedding100_32con.csv')
#
# # 提取数据
# X = data.iloc[:, :32].values  # 假设CSV中前32列是需要的数据
# # X_City = X[:6930, :32]        # 提取城市数据
# # X_GDP = X[90110:97040, :32]   # 提取GDP数据（注意Python索引从0开始）
# # X_In = X[159410:166340, :32]  # 提取其他数据
#
# # 将三个矩阵按行合并
# # X_data = np.vstack((X_City, X_GDP, X_In))  # 合并为 (6930*3) x 32 的矩阵
#
# # 使用 t-SNE 降维到三维
# tsne = TSNE(n_components=3, random_state=42)  # 设置随机种子以保证结果可重复
# Y = tsne.fit_transform(X)
# X_City = Y[:6930, :3]        # 提取城市数据
# X_GDP = Y[90110:97040, :3]   # 提取GDP数据（注意Python索引从0开始）
# X_In = Y[159410:166340, :3]  # 提取其他数据
#
#
# X_data = np.vstack((X_City, X_GDP, X_In))  # 合并为 (6930*3) x 32 的矩阵
#
# pd.DataFrame(X_data).to_csv('F:/新型区域活动要素知识图谱/Figure/Fig4/知识嵌入/Y_tsne_3d_back.csv', index=False)
#
# # 输出降维后的结果
# #print(Y.shape)  # 检查降维后的维度
#
# # # 可视化三维结果
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2])
# # ax.set_xlabel('Dimension 1')
# # ax.set_ylabel('Dimension 2')
# # ax.set_zlabel('Dimension 3')
# # ax.set_title('t-SNE 3D Visualization')
# # plt.show()