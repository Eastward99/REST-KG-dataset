import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F


def compute_similarity(embeddings):
    """
    计算节点嵌入之间的余弦相似度。
    embeddings: 节点嵌入矩阵，形状为 (num_nodes, embedding_dim)
    返回一个形状为 (num_nodes, num_nodes) 的相似度矩阵。
    """
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)  # 对嵌入进行L2归一化
    similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)
    return similarity_matrix



def graph_contrastive_loss(embeddings, edge_index, temperature=0.5, negative_samples_ratio=1.0):
    """
    A more memory-efficient version of the graph contrastive loss function.
    Instead of computing the full similarity matrix, we compute similarity only between neighbors.
    """
    # Calculate similarity between node embeddings for positive pairs
    positive_samples = []
    for i in range(edge_index.shape[1]):
        node_i = edge_index[0, i]
        node_j = edge_index[1, i]
        pos_sim = torch.matmul(embeddings[node_i], embeddings[node_j].T)
        positive_samples.append(pos_sim)

    # Convert to tensor
    positive_samples = torch.stack(positive_samples)

    # Number of positive samples
    num_positive_samples = positive_samples.size(0)

    # Calculate the number of negative samples we need
    num_negative_samples = int(num_positive_samples * negative_samples_ratio)

    # Negative sampling: generate random negative pairs
    node_i = torch.randint(0, embeddings.shape[0], (num_negative_samples,), device=embeddings.device)
    node_j = torch.randint(0, embeddings.shape[0], (num_negative_samples,), device=embeddings.device)

    # Ensure negative samples are distinct (i.e., node_i != node_j)
    mask = node_i == node_j
    while mask.sum() > 0:
        node_j[mask] = torch.randint(0, embeddings.shape[0], (mask.sum(),), device=embeddings.device)
        mask = node_i == node_j

    # Compute negative similarities
    negative_samples = torch.matmul(embeddings[node_i], embeddings[node_j].T)

    # Contrastive loss (simplified version)
    pos_sim = positive_samples / temperature
    neg_sim = negative_samples / temperature

    # Using log-sum-exp trick to avoid numerical instability
    loss = -torch.mean(torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)+ 1e-10)))

    return loss


def normalization(tensor):
    tensor_min = tensor.min(dim=0, keepdim=True).values
    tensor_max = tensor.max(dim=0, keepdim=True).values

    # Apply min-max normalization for each feature independently
    tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min)

    return tensor_normalized


def pad_features_to_position(features, start_dim, total_dim):
    current_dim = features.shape[1]
    padding_before = start_dim
    padding_after = total_dim - start_dim - current_dim
    return torch.cat([
        torch.zeros(features.shape[0], padding_before),  # Padding before
        features,  # Original features
        torch.zeros(features.shape[0], padding_after)  # Padding after
    ], dim=1)



def encoding_data(data):

    # 编码城市和省份名称
    province_encoder = LabelEncoder()
    data['province_id'] = province_encoder.fit_transform(data['所属省份'])

    city_encoder = LabelEncoder()
    time_encoder = LabelEncoder()
    # 土地指标
    UrbanArea_encoder = LabelEncoder()
    AdministrativeUnits_encoder = LabelEncoder()
    CulSpoFacilities_encoder = LabelEncoder()
    AgriculturalLand_encoder = LabelEncoder()

    # 人口指标
    PopSize_encoder = LabelEncoder()
    Practitioner_encoder = LabelEncoder()
    MedicalResources_encoder = LabelEncoder()
    SocialWelfare_encoder = LabelEncoder()
    CommunicationUsers_encoder = LabelEncoder()
    EducationalResources_encoder = LabelEncoder()

    #能源环境指标
    EnergyConsumption_encoder = LabelEncoder()
    EmissionsPollution_encoder = LabelEncoder()

    #经济指标
    GDP_encoder = LabelEncoder()
    IndustryAdded_encoder = LabelEncoder()
    IncomeSavings_encoder = LabelEncoder()
    FiscalTaxation_encoder = LabelEncoder()
    Exit_encoder = LabelEncoder()
    AgriculturalOutput_encoder = LabelEncoder()
    Investment_encoder = LabelEncoder()
    Consumption_encoder = LabelEncoder()
    Nightlight_encoder = LabelEncoder()
    Climate_encoder = LabelEncoder()

    year_long = 21

    data['city_id'] = city_encoder.fit_transform(data['所属城市'])
    data['time_id'] = time_encoder.fit_transform(data['年份'])+len(data['city_id'])
    #
    data['UrbanArea_id'] = UrbanArea_encoder.fit_transform(data['土地面积指标名称']) + len(data['city_id'])+year_long
    data['AdministrativeUnits_id'] = AdministrativeUnits_encoder.fit_transform(data['基层行政单位指标名称']) + 2*len(data['city_id']) + year_long
    data['CulSpoFacilities_id'] = CulSpoFacilities_encoder.fit_transform(data['文体设施指标名称']) + 3*len(data['city_id']) + year_long
    data['AgriculturalLand_id'] = AgriculturalLand_encoder.fit_transform(data['农业用地指标名称']) + 4* len(data['city_id']) + year_long
    #
    data['PopSize_id'] = PopSize_encoder.fit_transform(data['人口规模指标名称']) + 5 * len(data['city_id']) + year_long
    data['Practitioner_id'] = Practitioner_encoder.fit_transform(data['从业人员指标名称']) + 6*len(data['city_id']) + year_long
    data['MedicalResources_id'] = MedicalResources_encoder.fit_transform(data['医疗资源指标名称']) + 7*len(data['city_id']) + year_long
    data['SocialWelfare_id'] = SocialWelfare_encoder.fit_transform(data['社会福利指标名称']) + 8*len(data['city_id']) + year_long
    data['CommunicationUsers_id'] = CommunicationUsers_encoder.fit_transform(data['通讯用户指标名称']) + 9*len(data['city_id']) + year_long
    data['EducationalResources_id'] = EducationalResources_encoder.fit_transform(data['教育资源指标名称']) + 10*len(data['city_id']) + year_long
    #
    data['EnergyConsumption_id'] = EnergyConsumption_encoder.fit_transform(data['能源消耗指标名称']) + 11*len(data['city_id']) + year_long
    data['EmissionsPollution_id'] = EmissionsPollution_encoder.fit_transform(data['排放污染指标名称']) + 12*len(data['city_id']) + year_long
    #
    data['GDP_id'] = GDP_encoder.fit_transform(data['GDP指标名称']) + 13 * len(data['city_id']) +year_long
    data['IndustryAdded_1_id'] = IndustryAdded_encoder.fit_transform(data['第一产业增加值指标名称']) + 14 * len(data['city_id']) + year_long
    data['IndustryAdded_2_id'] = IndustryAdded_encoder.fit_transform(data['第二产业增加值指标名称']) + 15 * len(data['city_id']) + year_long
    data['IndustryAdded_3_id'] = IndustryAdded_encoder.fit_transform(data['第三产业增加值指标名称']) + 16 * len(data['city_id']) + year_long
    data['IndustryCon_id'] = IndustryAdded_encoder.fit_transform(data['产业结构指标名称']) + 17 * len(data['city_id']) + year_long


    data['IncomeSavings_id'] = IncomeSavings_encoder.fit_transform(data['收入储蓄指标名称']) + 18*len(data['city_id']) + year_long
    data['FiscalTaxation_id'] = FiscalTaxation_encoder.fit_transform(data['财政税收指标名称']) + 19*len(data['city_id']) + year_long
    data['FiscalExpend_id'] = FiscalTaxation_encoder.fit_transform(data['财政支出指标名称']) + 20 * len(data['city_id']) + year_long
    data['Exit_id'] = Exit_encoder.fit_transform(data['出口指标名称']) + 21*len(data['city_id']) + year_long
    data['AgriculturalOutput_id'] = AgriculturalOutput_encoder.fit_transform(data['农业产量指标名称']) + 22*len(data['city_id']) + year_long
    data['Investment_id'] = Investment_encoder.fit_transform(data['投资指标名称']) + 23*len(data['city_id']) + year_long
    data['Consumption_id'] = Consumption_encoder.fit_transform(data['消费指标名称']) + 24*len(data['city_id']) + year_long
    data['Nightlight_id'] = Nightlight_encoder.fit_transform(data['夜间灯光指标名称']) + 25 * len(data['city_id']) + year_long
    data['Climate_id'] = Climate_encoder.fit_transform(data['极端气候指标名称']) + 26 * len(data['city_id']) + year_long

    return data



def concat_edge(data,node_id,city_connection_path):
    node_id_map = {node_id.item(): idx for idx, node_id in enumerate(node_id)}

    # 城市-时间关系
    county_time_edges = torch.tensor([
        [node_id_map[time_id] for time_id in data['time_id'].values],
        [node_id_map[county_id] for county_id in data['city_id'].values]
    ], dtype=torch.long)

    # 城市-指标关系
    city_UrbanArea_edges = torch.tensor([[node_id_map[time_id] for time_id in data['UrbanArea_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]], dtype=torch.long)
    city_AdministrativeUnits_edges = torch.tensor([[node_id_map[time_id] for time_id in data['AdministrativeUnits_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_CulSpoFacilities_edges = torch.tensor([[node_id_map[time_id] for time_id in data['CulSpoFacilities_id'].values], [node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_AgriculturalLand_edges = torch.tensor([[node_id_map[time_id] for time_id in data['AgriculturalLand_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_PopSize_edges = torch.tensor([[node_id_map[time_id] for time_id in data['PopSize_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_Practitioner_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Practitioner_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_MedicalResources_edges = torch.tensor([[node_id_map[time_id] for time_id in data['MedicalResources_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_SocialWelfare_edges = torch.tensor([[node_id_map[time_id] for time_id in data['SocialWelfare_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_CommunicationUsers_edges = torch.tensor([[node_id_map[time_id] for time_id in data['CommunicationUsers_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_EducationalResources_edges = torch.tensor([[node_id_map[time_id] for time_id in data['EducationalResources_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_EnergyConsumption_edges = torch.tensor([[node_id_map[time_id] for time_id in data['EnergyConsumption_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_EmissionsPollution_edges = torch.tensor([[node_id_map[time_id] for time_id in data['EmissionsPollution_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['GDP_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)



    city_IndustryAdded_1_edges = torch.tensor([[node_id_map[time_id] for time_id in data['IndustryAdded_1_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_IndustryAdded_2_edges = torch.tensor([[node_id_map[time_id] for time_id in data['IndustryAdded_2_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_IndustryAdded_3_edges = torch.tensor([[node_id_map[time_id] for time_id in data['IndustryAdded_3_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_IndustryCon_edges = torch.tensor([[node_id_map[time_id] for time_id in data['IndustryCon_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)

    city_IncomeSavings_edges = torch.tensor([[node_id_map[time_id] for time_id in data['IncomeSavings_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_FiscalTaxation_edges = torch.tensor([[node_id_map[time_id] for time_id in data['FiscalTaxation_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_FiscalExpend_edges = torch.tensor([[node_id_map[time_id] for time_id in data['FiscalExpend_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)

    city_Exit_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Exit_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_AgriculturalOutput_edges = torch.tensor([[node_id_map[time_id] for time_id in data['AgriculturalOutput_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_Investment_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Investment_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_Consumption_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Consumption_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_Nightlight_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Nightlight_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)
    city_Climate_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Climate_id'].values],[node_id_map[city_id] for city_id in data['city_id'].values]],dtype=torch.long)

    city_indicator_edges = torch.cat([city_UrbanArea_edges,city_AdministrativeUnits_edges, city_CulSpoFacilities_edges,
                                      city_AgriculturalLand_edges, city_PopSize_edges, city_Practitioner_edges,
                                      city_MedicalResources_edges,  city_SocialWelfare_edges,city_CommunicationUsers_edges,
                                      city_EducationalResources_edges, city_EnergyConsumption_edges,city_EmissionsPollution_edges,
                                      city_GDP_edges, city_IndustryAdded_1_edges,city_IndustryAdded_2_edges,city_IndustryAdded_3_edges,city_IndustryCon_edges,
                                      city_IncomeSavings_edges,city_FiscalTaxation_edges, city_FiscalExpend_edges,city_Exit_edges,city_AgriculturalOutput_edges, city_Investment_edges,
                                      city_Consumption_edges,city_Nightlight_edges,city_Climate_edges
                                      ], dim=1)

    # 城市-城市空间关系#########################################################################################################
    Connfile_path = city_connection_path  # 替换为你的 CSV 文件路径
    connection_df = pd.read_csv(Connfile_path)
    years = range(2001, 2022)  # 从2000到2021，共22年

    # 创建一个空列表存储结果
    results = []

    # 遍历每一年，将对应年份添加到字符串数据
    for year in years:
        modified_df = connection_df.applymap(lambda x: f"{x}{year}" if isinstance(x, str) else x)
        results.append(modified_df)

    # 将所有年份的数据按行拼接
    final_df = pd.concat(results, ignore_index=True)
    name_to_id = pd.Series(data['city_id'].values, index=data['所属城市']).to_dict()

    final_df_replaced = final_df.applymap(lambda x: name_to_id.get(x, x))

    city_city_edges = torch.tensor([[node_id_map[join_id] for join_id in final_df_replaced['Join'].values], [node_id_map[target_id] for target_id in final_df_replaced['Target'].values]],dtype=torch.long)

    # 时间-时间关系#########################################################################################################
    data_filtered = [node_id_map[time_id] for time_id in data['time_id'].values]

    time_time_edges= torch.tensor([[time_id for time_id in range(min(data_filtered),max(data_filtered))],
                                    [time_id + 1  for time_id in range(min(data_filtered),max(data_filtered))]],dtype=torch.long)

    # 指标-指标—促进关系#########################################################################################################

    Consumption_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Consumption_id'].values],[node_id_map[city_id] for city_id in data['GDP_id'].values]],dtype=torch.long)
    Investment_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Investment_id'].values],[node_id_map[city_id] for city_id in data['GDP_id'].values]],dtype=torch.long)
    FiscalExpend_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['FiscalExpend_id'].values],[node_id_map[city_id] for city_id in data['GDP_id'].values]],dtype=torch.long)
    Popsize_Practitioner_edges = torch.tensor([[node_id_map[time_id] for time_id in data['PopSize_id'].values],[node_id_map[city_id] for city_id in data['Practitioner_id'].values]],dtype=torch.long)
    EducationalResources_Practitioner_edges = torch.tensor([[node_id_map[time_id] for time_id in data['EducationalResources_id'].values],[node_id_map[city_id] for city_id in data['Practitioner_id'].values]],dtype=torch.long)
    MedicalResources_Practitioner_edges = torch.tensor([[node_id_map[time_id] for time_id in data['MedicalResources_id'].values],[node_id_map[city_id] for city_id in data['Practitioner_id'].values]],dtype=torch.long)
    SocialWelfare_Practitioner_edges = torch.tensor([[node_id_map[time_id] for time_id in data['SocialWelfare_id'].values],[node_id_map[city_id] for city_id in data['Practitioner_id'].values]],dtype=torch.long)
    Practitioner_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Practitioner_id'].values],[node_id_map[city_id] for city_id in data['GDP_id'].values]],dtype=torch.long)
    IncomeSavings_Consumption_edges = torch.tensor([[node_id_map[time_id] for time_id in data['IncomeSavings_id'].values],[node_id_map[city_id] for city_id in data['Consumption_id'].values]], dtype=torch.long)
    UrbanArea_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['UrbanArea_id'].values],[node_id_map[city_id] for city_id in data['GDP_id'].values]], dtype=torch.long)


    IndustryAdded_1_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['IndustryAdded_1_id'].values],[node_id_map[city_id] for city_id in data['GDP_id'].values]],dtype=torch.long)
    IndustryAdded_2_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['IndustryAdded_2_id'].values],[node_id_map[city_id] for city_id in data['GDP_id'].values]], dtype=torch.long)
    IndustryAdded_3_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['IndustryAdded_3_id'].values],[node_id_map[city_id] for city_id in data['GDP_id'].values]],dtype=torch.long)

    PopSize_Nightlight_edges = torch.tensor([[node_id_map[time_id] for time_id in data['PopSize_id'].values],[node_id_map[city_id] for city_id in data['Nightlight_id'].values]],dtype=torch.long)
    AgriculturalLand_AgriculturalOutput_edges = torch.tensor([[node_id_map[time_id] for time_id in data['AgriculturalLand_id'].values],[node_id_map[city_id] for city_id in data['AgriculturalOutput_id'].values]],dtype=torch.long)
    AgriculturalOutput_IndustryAdded_3_edges = torch.tensor([[node_id_map[time_id] for time_id in data['AgriculturalOutput_id'].values],[node_id_map[city_id] for city_id in data['IndustryAdded_3_id'].values]],dtype=torch.long)

    # 指标-指标—抑制关系#########################################################################################################
    FiscalTaxation_Investment_edges = torch.tensor([[node_id_map[time_id] for time_id in data['FiscalTaxation_id'].values],[node_id_map[city_id] for city_id in data['Investment_id'].values]], dtype=torch.long)
    Climate_AgriculturalOutput_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Climate_id'].values],[node_id_map[city_id] for city_id in data['AgriculturalLand_id'].values]],dtype=torch.long)
    Climate_GDP_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Climate_id'].values],[node_id_map[city_id] for city_id in data['GDP_id'].values]],dtype=torch.long)
    Climate_Consumption_edges = torch.tensor([[node_id_map[time_id] for time_id in data['Climate_id'].values],[node_id_map[city_id] for city_id in data['Consumption_id'].values]],dtype=torch.long)

    # 合并 edge_index########################################################################################################
    indicator_city_edges = city_indicator_edges.flip(dims=[0])
    time_county_edges = county_time_edges.flip(dims=[0])

    edge_index = torch.cat([county_time_edges, city_indicator_edges,city_city_edges,time_time_edges,indicator_city_edges,time_county_edges,
                            Consumption_GDP_edges,Investment_GDP_edges,FiscalExpend_GDP_edges,Popsize_Practitioner_edges,EducationalResources_Practitioner_edges,
                            MedicalResources_Practitioner_edges,SocialWelfare_Practitioner_edges,Practitioner_GDP_edges,IncomeSavings_Consumption_edges,
                            UrbanArea_GDP_edges,IndustryAdded_1_GDP_edges,IndustryAdded_2_GDP_edges,IndustryAdded_3_GDP_edges,
                            PopSize_Nightlight_edges,AgriculturalLand_AgriculturalOutput_edges,AgriculturalOutput_IndustryAdded_3_edges,FiscalTaxation_Investment_edges,
                            Climate_AgriculturalOutput_edges,Climate_GDP_edges,Climate_Consumption_edges], dim=1)
    # edge_index = torch.cat([county_time_edges, city_indicator_edges,city_city_edges,time_time_edges,indicator_city_edges,time_county_edges], dim=1)


    # 使用 edge_type 记录每条边的关系类型#######################################################################################
    # 0 表示 has_time 关系，1 表示 has_indicator 关系
    edge_type = torch.cat([
        torch.zeros(county_time_edges.size(1), dtype=torch.long),  # has_time 关系
        torch.ones(city_indicator_edges.size(1), dtype=torch.long),  # has_indicator 关系
        torch.full((city_city_edges.size(1),), 2, dtype=torch.long),  # 指标节 点类型为 2
        torch.full((time_time_edges.size(1),), 3, dtype=torch.long),  # 指标节 点类型为 3
        torch.full((indicator_city_edges.size(1),), 4, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((time_county_edges.size(1),), 5, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((Consumption_GDP_edges.size(1),), 6, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((Investment_GDP_edges.size(1),), 7, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((FiscalExpend_GDP_edges.size(1),), 8, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((Popsize_Practitioner_edges.size(1),), 9, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((EducationalResources_Practitioner_edges.size(1),), 10, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((MedicalResources_Practitioner_edges.size(1),), 11, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((SocialWelfare_Practitioner_edges.size(1),), 12, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((Practitioner_GDP_edges.size(1),), 13, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((IncomeSavings_Consumption_edges.size(1),), 14, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((UrbanArea_GDP_edges.size(1),), 15, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((IndustryAdded_1_GDP_edges.size(1),), 16, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((IndustryAdded_2_GDP_edges.size(1),), 17, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((IndustryAdded_3_GDP_edges.size(1),), 18, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((PopSize_Nightlight_edges.size(1),), 19, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((AgriculturalLand_AgriculturalOutput_edges.size(1),), 20, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((AgriculturalOutput_IndustryAdded_3_edges.size(1),), 21, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((FiscalTaxation_Investment_edges.size(1),), 22, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((Climate_AgriculturalOutput_edges.size(1),), 23, dtype=torch.long),  # 指标节 点类型为 5
        torch.full((Climate_GDP_edges.size(1),), 24, dtype=torch.long),  # 指标节 点类型为 4
        torch.full((Climate_Consumption_edges.size(1),), 25, dtype=torch.long),  # 指标节 点类型为 5
    ])
    return node_id_map,edge_index,edge_type


def concat_node(data,city_features,time_features,UrbanArea_features,AdministrativeUnits_features,CulSpoFacilities_features,AgriculturalLand_features,PopSize_features,Practitioner_features,MedicalResources_features,SocialWelfare_features,
                CommunicationUsers_features,EducationalResources_features,EnergyConsumption_features,EmissionsPollution_features,GDP_features,IndustryAdded_1_features,IndustryAdded_2_features,IndustryAdded_3_features,IndustryCon_features,
                IncomeSavings_features,FiscalTaxation_features,FiscalExpend_features,Exit_features,AgriculturalOutput_features,Investment_features,Consumption_features,Nightlight_features,Climate_features):

    #normal
    city_features_normal = normalization(city_features)
    time_features_normal = normalization(time_features)
    UrbanArea_features_normal = normalization(UrbanArea_features)
    AdministrativeUnits_features_normal = normalization(AdministrativeUnits_features)
    CulSpoFacilities_features_normal = normalization(CulSpoFacilities_features)
    AgriculturalLand_features_normal = normalization(AgriculturalLand_features)
    PopSize_features_normal = normalization(PopSize_features)
    Practitioner_features_normal = normalization(Practitioner_features)
    MedicalResources_features_normal = normalization(MedicalResources_features)
    SocialWelfare_features_normal = normalization(SocialWelfare_features)
    CommunicationUsers_features_normal = normalization(CommunicationUsers_features)
    EducationalResources_features_normal = normalization(EducationalResources_features)
    EnergyConsumption_features_normal = normalization(EnergyConsumption_features)
    EmissionsPollution_features_normal = normalization(EmissionsPollution_features)
    GDP_features_normal = normalization(GDP_features)
    IndustryAdded_1_features_normal = normalization(IndustryAdded_1_features)
    IndustryAdded_2_features_normal = normalization(IndustryAdded_2_features)
    IndustryAdded_3_features_normal = normalization(IndustryAdded_3_features)
    IndustryCon_features_normal = normalization(IndustryCon_features)
    IncomeSavings_features_normal = normalization(IncomeSavings_features)
    FiscalTaxation_features_normal = normalization(FiscalTaxation_features)
    FiscalExpend_features_normal = normalization(FiscalExpend_features)
    Exit_features_normal = normalization(Exit_features)
    AgriculturalOutput_features_normal = normalization(AgriculturalOutput_features)
    Investment_features_normal = normalization(Investment_features)
    Consumption_features_normal = normalization(Consumption_features)
    Nightlight_features_normal = normalization(Nightlight_features)
    Climate_features_normal = normalization(Climate_features)


    feature_dimensions = [
        city_features_normal.shape[1],time_features_normal.shape[1],UrbanArea_features_normal.shape[1],
        AdministrativeUnits_features_normal.shape[1],CulSpoFacilities_features_normal.shape[1],AgriculturalLand_features_normal.shape[1],
        PopSize_features_normal.shape[1],Practitioner_features_normal.shape[1],MedicalResources_features_normal.shape[1],
        SocialWelfare_features_normal.shape[1],CommunicationUsers_features_normal.shape[1],EducationalResources_features_normal.shape[1],
        EnergyConsumption_features_normal.shape[1],EmissionsPollution_features_normal.shape[1],GDP_features_normal.shape[1],
        IndustryAdded_1_features_normal.shape[1],IndustryAdded_2_features_normal.shape[1],IndustryAdded_3_features_normal.shape[1],IndustryCon_features_normal.shape[1],
        IncomeSavings_features_normal.shape[1],FiscalTaxation_features_normal.shape[1],FiscalExpend_features_normal.shape[1],
        Exit_features_normal.shape[1],AgriculturalOutput_features_normal.shape[1],Investment_features_normal.shape[1],
        Consumption_features_normal.shape[1],Nightlight_features_normal.shape[1],Climate_features_normal.shape[1]]
    total_feature_dim = sum(feature_dimensions)

    start_positions = [0]
    for dim in feature_dimensions[:-1]:
        start_positions.append(start_positions[-1] + dim)

    city_features_padded = pad_features_to_position(city_features_normal, start_positions[0], total_feature_dim)
    time_features_padded = pad_features_to_position(time_features_normal, start_positions[1], total_feature_dim)
    UrbanArea_features_padded = pad_features_to_position(UrbanArea_features_normal, start_positions[2],total_feature_dim)
    AdministrativeUnits_features_padded = pad_features_to_position(AdministrativeUnits_features_normal,start_positions[3], total_feature_dim)
    CulSpoFacilities_features_padded = pad_features_to_position(CulSpoFacilities_features_normal, start_positions[4],total_feature_dim)
    AgriculturalLand_features_padded = pad_features_to_position(AgriculturalLand_features_normal, start_positions[5],total_feature_dim)
    PopSize_features_padded = pad_features_to_position(PopSize_features_normal, start_positions[6], total_feature_dim)
    Practitioner_features_padded = pad_features_to_position(Practitioner_features_normal, start_positions[7],total_feature_dim)
    MedicalResources_features_padded = pad_features_to_position(MedicalResources_features_normal, start_positions[8],total_feature_dim)
    SocialWelfare_features_padded = pad_features_to_position(SocialWelfare_features_normal, start_positions[9],total_feature_dim)
    CommunicationUsers_features_padded = pad_features_to_position(CommunicationUsers_features_normal,start_positions[10], total_feature_dim)
    EducationalResources_features_padded = pad_features_to_position(EducationalResources_features_normal,start_positions[11], total_feature_dim)
    EnergyConsumption_features_padded = pad_features_to_position(EnergyConsumption_features_normal, start_positions[12],total_feature_dim)
    EmissionsPollution_features_padded = pad_features_to_position(EmissionsPollution_features_normal,start_positions[13], total_feature_dim)
    GDP_features_padded = pad_features_to_position(GDP_features_normal, start_positions[14], total_feature_dim)
    IndustryAdded_1_features_padded = pad_features_to_position(IndustryAdded_1_features_normal, start_positions[15],total_feature_dim)
    IndustryAdded_2_features_padded = pad_features_to_position(IndustryAdded_2_features_normal, start_positions[16],total_feature_dim)
    IndustryAdded_3_features_padded = pad_features_to_position(IndustryAdded_3_features_normal, start_positions[17],total_feature_dim)
    IndustryCon_features_padded = pad_features_to_position(IndustryCon_features_normal, start_positions[18],total_feature_dim)

    IncomeSavings_features_padded = pad_features_to_position(IncomeSavings_features_normal, start_positions[19],total_feature_dim)
    FiscalExpend_features_padded = pad_features_to_position(FiscalExpend_features_normal, start_positions[20],total_feature_dim)
    FiscalTaxation_features_padded = pad_features_to_position(FiscalTaxation_features_normal, start_positions[21],total_feature_dim)


    Exit_features_padded = pad_features_to_position(Exit_features_normal, start_positions[22], total_feature_dim)
    AgriculturalOutput_features_padded = pad_features_to_position(AgriculturalOutput_features_normal,start_positions[23], total_feature_dim)
    Investment_features_padded = pad_features_to_position(Investment_features_normal, start_positions[24],total_feature_dim)
    Consumption_features_padded = pad_features_to_position(Consumption_features_normal, start_positions[25], total_feature_dim)
    Nightlight_features_padded = pad_features_to_position(Nightlight_features_normal, start_positions[26], total_feature_dim)
    Climate_features_padded = pad_features_to_position(Climate_features_normal, start_positions[27], total_feature_dim)




    # Step 1: 合并所有节点特征到 node_features 中
    node_features = torch.cat((city_features_padded,time_features_padded,UrbanArea_features_padded,AdministrativeUnits_features_padded,
        CulSpoFacilities_features_padded,AgriculturalLand_features_padded,PopSize_features_padded,Practitioner_features_padded,
        MedicalResources_features_padded,SocialWelfare_features_padded,CommunicationUsers_features_padded,EducationalResources_features_padded,
        EnergyConsumption_features_padded,EmissionsPollution_features_padded,GDP_features_padded,IndustryAdded_1_features_padded,
        IndustryAdded_2_features_padded,IndustryAdded_3_features_padded,IndustryCon_features_padded,IncomeSavings_features_padded,
        FiscalTaxation_features_padded,FiscalExpend_features_padded,Exit_features_padded,AgriculturalOutput_features_padded,
        Investment_features_padded,Consumption_features_padded,Nightlight_features_padded,Climate_features_padded
    ), dim=0)  # Concatenate along node dimension

    node_id1 = torch.cat([torch.tensor(data['city_id'].values)]).unique()
    node_id2 = torch.cat([torch.tensor(data['time_id'].values)]).unique()
    node_id3 = torch.cat([torch.tensor(data['UrbanArea_id'].values)]).unique()
    node_id4 = torch.cat([torch.tensor(data['AdministrativeUnits_id'].values)]).unique()
    node_id5 = torch.cat([torch.tensor(data['CulSpoFacilities_id'].values)]).unique()
    node_id6 = torch.cat([torch.tensor(data['AgriculturalLand_id'].values)]).unique()
    node_id7 = torch.cat([torch.tensor(data['PopSize_id'].values)]).unique()
    node_id8 = torch.cat([torch.tensor(data['Practitioner_id'].values)]).unique()
    node_id9 = torch.cat([torch.tensor(data['MedicalResources_id'].values)]).unique()
    node_id10 = torch.cat([torch.tensor(data['SocialWelfare_id'].values)]).unique()
    node_id11 = torch.cat([torch.tensor(data['CommunicationUsers_id'].values)]).unique()
    node_id12 = torch.cat([torch.tensor(data['EducationalResources_id'].values)]).unique()
    node_id13 = torch.cat([torch.tensor(data['EnergyConsumption_id'].values)]).unique()
    node_id14 = torch.cat([torch.tensor(data['EmissionsPollution_id'].values)]).unique()
    node_id15 = torch.cat([torch.tensor(data['GDP_id'].values)]).unique()
    node_id16 = torch.cat([torch.tensor(data['IndustryAdded_1_id'].values)]).unique()
    node_id17 = torch.cat([torch.tensor(data['IndustryAdded_2_id'].values)]).unique()
    node_id18 = torch.cat([torch.tensor(data['IndustryAdded_3_id'].values)]).unique()
    node_id19 = torch.cat([torch.tensor(data['IndustryCon_id'].values)]).unique()

    node_id20 = torch.cat([torch.tensor(data['IncomeSavings_id'].values)]).unique()
    node_id21 = torch.cat([torch.tensor(data['FiscalTaxation_id'].values)]).unique()
    node_id22 = torch.cat([torch.tensor(data['FiscalExpend_id'].values)]).unique()
    node_id23 = torch.cat([torch.tensor(data['Exit_id'].values)]).unique()
    node_id24 = torch.cat([torch.tensor(data['AgriculturalOutput_id'].values)]).unique()
    node_id25 = torch.cat([torch.tensor(data['Investment_id'].values)]).unique()
    node_id26 = torch.cat([torch.tensor(data['Consumption_id'].values)]).unique()
    node_id27 = torch.cat([torch.tensor(data['Nightlight_id'].values)]).unique()
    node_id28 = torch.cat([torch.tensor(data['Climate_id'].values)]).unique()


    node_id = torch.cat([node_id1,node_id2,node_id3,node_id4,node_id5,node_id6,
                         node_id7,node_id8,node_id9,node_id10,node_id11,node_id12,
                         node_id13, node_id14, node_id15,node_id16,node_id17,node_id18,
                         node_id19, node_id20, node_id21,node_id22,node_id23,node_id24,
                         node_id25,node_id26,node_id27,node_id28])


    # Step 2: 使用 node_type 记录节点类型
    # 0 表示城市节点，1 表示时间节点，2 表示指标节点
    node_type = torch.cat([
        torch.zeros(city_features.size(0), dtype=torch.long),       # 城市节点类型为 0
        torch.ones(time_features.size(0), dtype=torch.long),        # 时间节点类型为 1
        torch.full((UrbanArea_features.size(0),), 2, dtype=torch.long), # 指标节 点类型为 2
        torch.full((AdministrativeUnits_features.size(0),), 3, dtype=torch.long),  # 指标节 点
        torch.full((CulSpoFacilities_features.size(0),), 4, dtype=torch.long),  # 指标节 点
        torch.full((AgriculturalLand_features.size(0),), 5, dtype=torch.long),  # 指标节 点
        torch.full((PopSize_features.size(0),), 6, dtype=torch.long),  # 指标节 点
        torch.full((Practitioner_features.size(0),), 7, dtype=torch.long),  # 指标节 点
        torch.full((MedicalResources_features.size(0),), 8, dtype=torch.long),  # 指标节 点
        torch.full((SocialWelfare_features.size(0),), 9, dtype=torch.long),  # 指标节 点
        torch.full((CommunicationUsers_features.size(0),), 10, dtype=torch.long),  # 指标节 点
        torch.full((EducationalResources_features.size(0),), 11, dtype=torch.long),  # 指标节 点
        torch.full((EnergyConsumption_features.size(0),), 12, dtype=torch.long),  # 指标节 点
        torch.full((EmissionsPollution_features.size(0),), 13, dtype=torch.long),  # 指标节 点
        torch.full((GDP_features.size(0),), 14, dtype=torch.long),  # 指标节 点
        torch.full((IndustryAdded_1_features.size(0),), 15, dtype=torch.long),  # 指标节 点
        torch.full((IndustryAdded_2_features.size(0),), 16, dtype=torch.long),  # 指标节 点
        torch.full((IndustryAdded_3_features.size(0),), 17, dtype=torch.long),  # 指标节 点
        torch.full((IndustryAdded_3_features.size(0),), 18, dtype=torch.long),  # 指标节 点

        torch.full((IncomeSavings_features.size(0),), 19, dtype=torch.long),  # 指标节 点
        torch.full((FiscalTaxation_features.size(0),), 20, dtype=torch.long),  # 指标节 点
        torch.full((FiscalExpend_features.size(0),), 21, dtype=torch.long),  # 指标节 点

        torch.full((Exit_features.size(0),), 22, dtype=torch.long),  # 指标节 点
        torch.full((AgriculturalOutput_features.size(0),), 23, dtype=torch.long),  # 指标节 点
        torch.full((Investment_features.size(0),), 24, dtype=torch.long),  # 指标节 点
        torch.full((Consumption_features.size(0),), 25, dtype=torch.long),  # 指标节 点
        torch.full((Nightlight_features.size(0),), 26, dtype=torch.long),  # 指标节 点
        torch.full((Climate_features.size(0),), 27, dtype=torch.long)  # 指标节 点
    ])

    return node_features,node_id,node_type,total_feature_dim

def graph_make(data_path):
    data = pd.read_csv(data_path)


    data = encoding_data(data)
    # 获取城市属性特征（经纬度、所属省份等）
    city_features = torch.tensor(data[['经度', '纬度','行政区域土地面积(平方公里)']].values,dtype=torch.float)

    # 时间实体特征（可以是年份）
    time_features = torch.tensor(data['年份'].unique(),dtype=torch.float).view(-1, 1)  # 转换成二维矩阵

    # 指标实体特征（例如土地面积、乡镇数等）R
    UrbanArea_features = torch.tensor(data[['建成区面积(平方公里)']].values,dtype=torch.float)
    AdministrativeUnits_features = torch.tensor(data[['乡及镇个数(个)',  '街道办事处个数(个)', '村民委员会个数(个)']].values,dtype=torch.float)
    CulSpoFacilities_features = torch.tensor(data[['艺术表演场馆数_剧场、影剧院(个)', '公共图书馆总藏量(千册)', '体育场馆机构数(个)']].values,dtype=torch.float)
    AgriculturalLand_features = torch.tensor(data[['农作物总播种面积(千公顷)', '常用耕地面积(公顷)', '机收面积(公顷)']].values,dtype=torch.float)

    # 人口指标
    PopSize_features = torch.tensor(data[['年末总户数(户)', '年末总人口(万人)', '户籍人口数(万人)']].values,dtype=torch.float)
    Practitioner_features = torch.tensor(data[['年末单位从业人员(人)', '城镇单位在岗职工人数(人)', '乡村从业人员数(人)', '农林牧渔业从业人员数(人)', '年末第二产业单位从业人员(人)'
                                               , '年末第三产业单位从业人员(人)']].values,dtype=torch.float)
    MedicalResources_features = torch.tensor(data[['医院、卫生院床位数(床)', '医院和卫生院卫生人员数_卫生技术人员(人)', '医院和卫生院卫生人员数_执业医师(人)']].values,dtype=torch.float)
    SocialWelfare_features = torch.tensor(data[['各种社会福利收养性单位数(个)', '各种社会福利收养性单位床位数(床)']].values,dtype=torch.float)
    CommunicationUsers_features = torch.tensor(data[['固定电话用户(户)', '移动电话用户数(户)', '宽带接入用户数(户)']].values,dtype=torch.float)
    EducationalResources_features = torch.tensor(data[['普通小学学校数(个)', '普通中学学校数(个)', '普通小学专任教师数(人)', '普通中学专任教师数(人)', '普通小学在校生数(人)',
                                                       '普通中学在校学生数(人)','中等职业教育学校在校学生数']].values,dtype=torch.float)

    # 能源环境指标
    EnergyConsumption_features = torch.tensor(data[['农用机械总动力(千万瓦)', '全社会用电量(万千瓦时)', '城乡居民生活用电量(万千瓦时)']].values,dtype=torch.float)
    EmissionsPollution_features = torch.tensor(data[['废气中氮氧化物排放量(吨)', '废气中烟尘排放量(吨)', '工业废气中二氧化硫排放量(吨)']].values,dtype=torch.float)

    # 经济指标
    GDP_features = torch.tensor(data[['地区生产总值(万元)', 'GDP增量','GDP增速']].values,dtype=torch.float)
    IndustryAdded_1_features = torch.tensor(data[['第一产业增加值(万元)', '第一产业增加量(万元)', '第一产业增速']].values,dtype=torch.float)
    IndustryAdded_2_features = torch.tensor(data[['第二产业增加值(万元)', '第二产业增加量(万元)', '第二产业增速']].values,dtype=torch.float)
    IndustryAdded_3_features = torch.tensor(data[['第三产业增加值(万元)', '第三产业增加量(万元)', '第三产业增速']].values,dtype=torch.float)
    IndustryCon_features = torch.tensor(data[['主要产业']].values,dtype=torch.float)
    IncomeSavings_features = torch.tensor(data[['城镇单位在岗职工总工资(元)', '城乡居民储蓄存款余额(万元)', '年末金融机构各项贷款余额(万元)']].values,dtype=torch.float)
    FiscalTaxation_features = torch.tensor(data[['地方财政一般预算收入(万元)', '各项税收(万元)']].values,dtype=torch.float)
    FiscalExpend_features= torch.tensor(data[['地方财政一般预算支出(万元)']].values,dtype=torch.float)
    Exit_features = torch.tensor(data[['出口额(美元)', '实际利用外资金额(美元)']].values,dtype=torch.float)
    AgriculturalOutput_features = torch.tensor(data[['粮食总产量(吨)', '棉花产量(吨)', '油料产量(吨)', '肉类总产量(吨)']].values,dtype=torch.float)
    Investment_features = torch.tensor(data[['城镇固定资产投资完成额(万元)', '全社会固定资产投资(万元)', '房地产开发投资(亿元)']].values,dtype=torch.float)
    Consumption_features = torch.tensor(data[['社会消费品零售总额(万元)']].values,dtype=torch.float)
    Nightlight_features = torch.tensor(data[['夜间灯光平均强度']].values,dtype=torch.float)
    Climate_features = torch.tensor(data[['极端降雨', '极端高温']].values,dtype=torch.float)

    # 实体总和编码
    node_features, node_id, node_type, total_feature_dim = concat_node(data,city_features,time_features,UrbanArea_features,AdministrativeUnits_features,CulSpoFacilities_features,AgriculturalLand_features,PopSize_features,Practitioner_features,MedicalResources_features,SocialWelfare_features,CommunicationUsers_features,EducationalResources_features,
    EnergyConsumption_features,EmissionsPollution_features,GDP_features,IndustryAdded_1_features,IndustryAdded_2_features,IndustryAdded_3_features,IndustryCon_features,IncomeSavings_features,FiscalTaxation_features,FiscalExpend_features,Exit_features,AgriculturalOutput_features,
    Investment_features,Consumption_features,Nightlight_features,Climate_features)

    print("nodeconcat")
    ###################################################################################################################################################
    # 创建异构图
    # Step 3: 构建 edge_index 和 edge_type
    # 构建城市到时间的关系边（例如年份关系）
    node_id_map, edge_index, edge_type = concat_edge(data,node_id,'./data/空间关系/城市相邻关系_替换后.csv')

    return data, node_features,node_id,node_type,node_id_map, edge_index, edge_type, total_feature_dim

if __name__ == '__main__':
# #
    data, node_features,node_id,node_type,node_id_map, edge_index, edge_type,total_feature_dim = graph_make('./data/grouped_data_filtered0121.csv')

    print('end')