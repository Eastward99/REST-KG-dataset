import numpy as np
import pandas as pd

# 读取CSV文件
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def datapro():
    file_path = './data/县级.csv'  # 替换为实际的CSV文件路径
    df = pd.read_csv(file_path)

    grouped_df = df.groupby(['年份', '所属城市']).sum().reset_index()



    # 保存整合后的数据到新的CSV文件
    grouped_df.to_csv('./data/grouped_data.csv', index=False)

    print("整合后的数据已保存为 grouped_data.csv")


def xiugai():
    frist_path = './data/县级.csv'  # 替换为实际的CSV文件路径
    file_path = './data/grouped_data.csv'  # 替换为实际的CSV文件路径


    df0 = pd.read_csv(frist_path)
    df1 = pd.read_csv(file_path)

    for index, row in df1.iterrows():
        value_to_find = row[1]

        first_occurrence = df0[df0['所属城市'] == value_to_find]

        # 获取第一次出现的行索引
        if not first_occurrence.empty:
            first_row_index = first_occurrence.index[0]
            df1.at[index, '行政区划代码'] = df0.at[first_row_index,'行政区划代码']
            df1.at[index, '所属省份'] = df0.at[first_row_index, '所属省份']
            df1.at[index, '经度'] = df0.at[first_row_index,'经度']
            df1.at[index, '纬度'] = df0.at[first_row_index,'纬度']



            #print(f"'{value_to_find}' 在 '所属城市' 列中第一次出现的行索引为: {first_row_index}")
        else:
            print(f"在 '所属城市' 列中未找到 '{value_to_find}' 的匹配值")

    df1.to_csv('./data/grouped_data_final.csv', index=False)
    print("整合后的数据已保存为 grouped_data_final.csv")

    return 0



#
# 读取 CSV 文件
def shujutihuan():
    file_path = 'F:/新型区域活动要素知识图谱/极端气候指标/RESLUT_FD0.csv'  # 替换为你的 CSV 文件路径
    df = pd.read_csv(file_path)
    # years = range(2000, 2021)  # 从2000到2021，共22年
    #
    # # 创建一个空列表存储结果
    # results = []
    #
    # # 遍历每一年，将对应年份添加到字符串数据
    # for year in years:
    #     modified_df = df.applymap(lambda x: f"{x}{year}" if isinstance(x, str) else x)
    #     results.append(modified_df)
    #
    # # 将所有年份的数据按行拼接
    # final_df = pd.concat(results, ignore_index=True)
    #
    file_b_path = 'F:/新型区域活动要素知识图谱/极端气候指标/stations.csv'
    df_b = pd.read_csv(file_b_path)


    # 创建映射字典
    name_to_id = pd.Series(df_b['市'].values, index=df_b['stationid']).to_dict()

    # 替换中文为id
    df_a_replaced = df.applymap(lambda x: name_to_id.get(x, x))

    # 保存结果
    output_file_path = 'F:/新型区域活动要素知识图谱/极端气候指标/file_a_replaced_FD0.csv'
    df_a_replaced.to_csv(output_file_path, index=False)

    print(f"替换完成，结果已保存到 {output_file_path}")

# import pandas as pd
#
# # 读取原始 CSV 文件
# df = pd.read_csv('F:/新型区域活动要素知识图谱/极端气候指标/file_a_replaced_FD0_weiyi.csv')
#
# # 将年份列设置为索引
# df.set_index('station', inplace=True)
#
# # 转置数据，将行数据转换为列数据
# df_transposed = df.stack().reset_index()
#
# # 设置列名
# df_transposed.columns = ['列名','年份', '值']
#
#
# # 保存结果到新的 CSV 文件
# df_transposed.to_csv('F:/新型区域活动要素知识图谱/极端气候指标/Climate_data_FD0.csv', index=False)


# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv('F:/新型区域活动要素知识图谱/极端气候指标/file_a_replaced_FD0.csv')
# # 按照 'column_name' 列分组，并计算其他列的平均值
# df_grouped = df.groupby('station').agg('mean').reset_index()
# df_grouped.to_csv('F:/新型区域活动要素知识图谱/极端气候指标/file_a_replaced_FD0_weiyi.csv', index=False)


# 打印结果查看
# print(df_transposed)
# #
# import os
# import pandas as pd
#
# # 文件夹路径
# folder_path = 'F:/新型区域活动要素知识图谱/极端气候指标/FD0'
#
# # 初始化一个空的 DataFrame 来存储合并后的数据
# merged_df = pd.DataFrame()
#
# # 获取文件夹中所有的 CSV 文件
# csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
#
# # 处理每个 CSV 文件
# # 处理每个 CSV 文件
# for file in csv_files:
#     # 读取每个 CSV 文件
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_csv(file_path)
#
#     # 筛选第一列数值在 1999 到 2020 之间的行
#     filtered_df = df[(df.iloc[:, 0] >= 1999) & (df.iloc[:, 0] <= 2020)]
#
#     # 如果筛选后有数据，提取第二列数据
#     if not filtered_df.empty:
#         second_column = filtered_df.iloc[:, 1].reset_index(drop=True)
#
#         # 如果 merged_df 为空，初始化其第一列
#         if merged_df.empty:
#             merged_df[file] = second_column
#         else:
#             # 合并数据，保证数据对齐
#             merged_df = pd.concat([merged_df, second_column], axis=1, ignore_index=False)
#
# # 将文件名作为第一行
# merged_df.columns = csv_files
#
# # 将结果保存到一个新的 CSV 文件
# merged_df.to_csv('F:/新型区域活动要素知识图谱/极端气候指标/RESLUT_FD0.csv', index=False)


# import pandas as pd
#
# # 读取csv文件
# csv1 = pd.read_csv('F:/新型区域活动要素知识图谱/极端气候指标/Climate_data_FD0.csv')  # 包含年份、城市、数值
# csv2 = pd.read_csv('F:/新型区域活动要素知识图谱/极端气候指标/辅助计算.csv')  # 目标文件，包含年份、城市和待填充的数值
#
# # 合并两个DataFrame，按'年份'和'城市'进行匹配
# merged_csv = pd.merge(csv2, csv1[['年份', '城市', '数值']], on=['年份', '城市'], how='left')
#
# # 将合并后的结果保存为新的csv文件
# merged_csv.to_csv('F:/新型区域活动要素知识图谱/极端气候指标/merged_FD0.csv', index=False)

def load_csv_to_numpy(csv_file_path):
    # 读取CSV文件为DataFrame
    df = pd.read_csv(csv_file_path)

    # 将DataFrame转换为numpy数组
    data = df.to_numpy()  # 或者使用 df.values

    return data

def tsne_dimensionality_reduction(X, labels=None, n_components=2, perplexity=30, learning_rate=200, n_iter=1000,
                              random_state=None, plot=True):
    """
使用t-SNE将高维数据降至低维空间。
    """
    # 初始化t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate,
                n_iter=n_iter, random_state=random_state)

    # 进行降维
    X_2D = tsne.fit_transform(X)

    # 如果plot为True，绘制降维后的数据
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_2D[:, 0], X_2D[:, 1], c='blue', s=30, alpha=0.7)
        plt.title("t-SNE Dimensionality Reduction")
        plt.show()

    return X_2D


def kmeans_clustering(X_2D, n_clusters=3):
    """
    使用KMeans对降维后的数据进行聚类。
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_2D)

    return labels, kmeans


def plot_kmeans_clusters(X_2D, labels):
    """
    绘制KMeans聚类的结果。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels, cmap='jet', s=30, alpha=0.7)
    plt.colorbar()  # 显示颜色条
    plt.title("KMeans Clustering on t-SNE Reduced Data")
    plt.show()

csv_file_path = 'F:/新型区域活动要素知识图谱/Figure/Fig4/知识嵌入/GDP32_Embeddingcsv.csv'

# 调用函数加载数据
data = load_csv_to_numpy(csv_file_path)


# 进行t-SNE降维
X_2D = tsne_dimensionality_reduction(data, plot=True)
#X_2D_1 = X_2D[1::21, :]
# 使用KMeans聚类
n_clusters = 5  # 设定聚类数，根据需要调整
labels, kmeans = kmeans_clustering(X_2D, n_clusters=n_clusters)

# 绘制KMeans聚类结果
plot_kmeans_clusters(X_2D, labels)
print("end")



