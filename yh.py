import os
from collections import Counter
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from utils import feature_selection_with_lasso
import shap
from utils import evaluate_performance, get_longitudinal




data_before, data_after, feat_names = get_longitudinal()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# pr = pd.DataFrame(columns=['id', 0, 1, 2, 3, 4])
pr = pd.DataFrame(columns=['id', 'Score'])

class TT(nn.Module):
    def __init__(self, input_dim, transformer_dim, num_heads, num_layers, hidden_dim, output_dim):
        super(TT, self).__init__()

        class Permute(nn.Module):
            def __init__(self, *dims):
                super(Permute, self).__init__()
                self.dims = dims

            def forward(self, x):
                return x.permute(self.dims)
        # self.fc1 = nn.Linear(input_dim+2, transformer_dim)  # 特征维度转换
        # self.fc1 = nn.Linear(1037, transformer_dim)  # 特征维度转换
        encoder_layers = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads,
                                                    dim_feedforward=hidden_dim)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.features = nn.Sequential(
            nn.Linear(1037, transformer_dim),
            Permute(1, 0, 2),
            nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        )

        self.fc2 = nn.Linear(512, output_dim)  # 预测层
    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=0)
        feat = x
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x, feat
class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        # 定义一个全连接层
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc(x)

        # x = torch.sigmoid(x)
        return x
class NG(nn.Module):
    def __init__(self, input_size):
        super(NG, self).__init__()
        # 定义一个全连接层
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(input_size-512, 64)
        self.fc3 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
    def forward(self, x, epoch=50):
        xx = x[:, 512:]
        x = x[:,:512]
        x = self.fc1(x)
        x = self.dropout1(x)
        xx = self.fc2(xx)
        xx = self.dropout1(xx)
        x = torch.cat((x, xx), dim=1)
        x = self.fc3(x)
        x = self.dropout2(x)
        return x
class GG(nn.Module):
    def __init__(self, input_size):
        super(GG, self).__init__()
        # 定义一个全连接层
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(23, 64)
        self.fc3 = nn.Linear(23, 64)
        self.Longitudinal_Sequence = nn.Sequential(
            nn.RNN(64, 64, 1, batch_first=True),
        )
        self.fc = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(1)
        self.dropout2 = nn.Dropout(0.5)
    def forward(self, x, epoch=50):
        x1 = x[:, self.input_size:self.input_size+23]
        x2 = x[:, self.input_size+23:self.input_size+46]
        x = x[:, :self.input_size]
        # x = torch.cat((x, m), dim=1)
        x = self.fc1(x)
        if epoch < 50:
            x = self.dropout1(x)
        else:
            x = self.dropout2(x)
        x1 = self.fc2(x1)
        x2 = self.fc3(x2)
        x3 = torch.stack([x1, x2], dim=1)  # 将特征序列排列在时间维度上
        x3 = torch.squeeze(x3, -1).float()
        _, x_sequence = self.Longitudinal_Sequence(x3)
        x3 = x_sequence[-1, :, :]
        x = torch.cat((x, x3), dim=1)
        x= self.fc(x)
        return x

def get_cancer(features_list):
    # print(len(features_list))
    # model = NN(1024).to(device)
    model = torch.load('./is_cancer_20x.pth', map_location=device)
    # model.load_state_dict(static_dict)
    model.eval()
    output_list = []
    with torch.no_grad():  # 在评估模式下不计算梯度
        for i in range(0, len(features_list), 2048):
            # 获取当前批次
            batch = features_list[i:i + 2048]
            # 将批次数据转换为张量
            batch_tensor = torch.stack(batch).to(device)
            # 计算模型输出
            output = model(batch_tensor)
            # 将输出添加到输出列表中

            # print(output.tolist())
            output_list.extend(output.tolist())

    # 将输出与索引配对并根据输出从大到小排序
    indexed_output = list(enumerate(output_list))
    indexed_output.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, value in indexed_output]
    # print(sorted_indices)
    return sorted_indices
class Load(Dataset):

    def __init__(self,files_path, coords_path):
        print('文件路径：', files_path)
        self.names = []
        self.data = {}
        self.n = []
        coords = torch.load(coords_path, map_location='cpu')
        rad_feat = 'ImmunotherapyCT_feature.csv'
        df = pd.read_csv(rad_feat)
        df_mean = []

        # Standardize each column except for the ID column
        for col in df.columns:
            if col != 'id':  # Assuming 'id' is the name of your ID column
                df_mean.append(df[col].mean())
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        for i in tqdm(os.listdir(files_path)):
            name = i[:-3]

            l, r = coords[name]

            q = torch.load(os.path.join(files_path, i), map_location='cpu')
            features = []
            coordinates = []
            for (x, y), tensor in q.items():
                features.append(tensor)
                coordinates.append([int(x) / int(l) , int(y) / int(r)])
            sorted_index = get_cancer(features)
            features = [features[index] for index in sorted_index][:len(features)//7 * 7]
            coordinates = [coordinates[index] for index in sorted_index][:len(coordinates)//7 * 7]

            # features = torch.tensor([item.cpu().detach().numpy() for item in features])
            # if len(features) < 1000:
            #     continue
            features_tensor = torch.stack(features).squeeze()
            #
            mean = features_tensor.mean(dim=1, keepdim=True)
            std = features_tensor.std(dim=1, keepdim=True)
            features_tensor = (features_tensor-mean)/std
            # print(mean)
            coordinates_tensor = torch.tensor(coordinates)
            # combined_tensor = torch.cat((coordinates_tensor, features_tensor), dim=1)

            row = df[df['id'] == name]

            row_tensor = torch.tensor(row.drop('id', axis=1).values)
            if row_tensor.shape[0] == 0:
                continue
                row_tensor = torch.tensor(df_mean)

            combined_tensor = torch.cat((features_tensor, coordinates_tensor), dim=1)
            n = combined_tensor.shape[0]
            row_tensor = row_tensor.repeat(n, 1)

            # print(row_tensor)
            # print(combined_tensor.shape)
            indices = torch.randperm(n)[:]
            selected_tensor = combined_tensor[indices]
            sort_key = selected_tensor[:, -2] + selected_tensor[:, -1] / 1000000
            # 根据组合键排序
            sorted_indices = torch.argsort(sort_key)
            sorted_tensor = selected_tensor[sorted_indices]
            combined_tensor = torch.cat((sorted_tensor, row_tensor), dim=1)

            self.names.append(name)

            self.data[name] = combined_tensor[:].to(torch.float32)



        print(len(self.names))


    def __len__(self):
        # return len(self.names)
        return len(self.names)

    def __getitem__(self, item) -> tuple:
        name = self.names[item]

        return name, self.data[name][:, :]
    def res_names(self):
        return self.names

def get_data(data_loader, model_path, csv_path,  i,clinic_path=None):
    scaler = StandardScaler()
    X1 = data_before
    X2 = data_after
    scaler = StandardScaler()
    global feat_names
    def standardize_features():
        # Check if feature names are in both dataframes
        for feature in feat_names:
            if feature not in X1.columns or feature not in X2.columns:
                raise ValueError(f"Feature '{feature}' is not present in both datasets")

        # Extract only the columns to be standardized
        features_before = X1[feat_names]
        features_after = X2[feat_names]

        # Standardize the features
        scaler.fit(features_before)
        standardized_features_before = scaler.transform(features_before)
        standardized_features_after = scaler.transform(features_after)

        # Replace the columns in the original dataframes with the standardized values
        X1[feat_names] = standardized_features_before
        X2[feat_names] = standardized_features_after

        return X1, X2
    # X1, X2 = standardize_features()
    data1 = []
    data2 = []
    csv_file = pd.read_excel(csv_path)

    csv_file = csv_file[csv_file.iloc[:, 1] != 'PD']
    csv_file.iloc[:, 1] = csv_file.iloc[:, 1].apply(lambda x: 1 if x == 'SD' else 0)
    # csv_file.iloc[:, 3] = csv_file.iloc[:, 3].apply(lambda x: 1 if x < 12 else 0)
    # csv_file.iloc[:, 3] = csv_file.apply(
    #     lambda row: 0 if row.iloc[7] == 0 else row.iloc[3], axis=1)

    print(csv_file)
    clinic_csv = pd.read_excel(clinic_path)
    # try:
    #     clinic_csv = pd.read_excel(clinic_path, encoding='latin1')  # 尝试使用 'latin1' 编码读取
    # except UnicodeDecodeError:
    #     try:
    #         clinic_csv = pd.read_excel(clinic_path, encoding='ISO-8859-1')  # 尝试使用 'ISO-8859-1' 编码读取
    #     except Exception as e:
    #         print("Error reading the CSV file:", e)
    #         return None, None


    data = []
    labels = []
    outputs = []
    clinic_feat = []
    timetime = []

    names = []
    mean_values = clinic_csv.mean(numeric_only=True)
    clinic_csv = clinic_csv.fillna(mean_values)
    df = clinic_csv
    df = pd.read_csv('rad_raw.csv')
    df = pd.merge(clinic_csv, df, on='id')
    df = feature_selection_with_lasso(df, csv_file[['id', 'BOR']], 0.05)

    feature_names = list(range(1, 513))
    feature_names = [f"H&E {num}" for num in feature_names]
    feature_names.extend(list(df.drop(columns=['id']).columns))
    # clinic_csv = feature_selection_with_lasso(clinic_csv, csv_file, np.logspace(-4, -4, 100))
    # df = feature_selection_with_lasso(df, csv_file)
    # print(df)
    input_dim = 1037
    transformer_dim = 512
    num_heads = 8
    num_layers = 2
    hidden_dim = 2048
    output_dim = 1  # 假设我们预测一个标量值
    model = TT(input_dim, transformer_dim, num_heads, num_layers, hidden_dim, output_dim).to(device)
    static_dict = torch.load(model_path)
    model.load_state_dict(static_dict)
    model.eval()
    results = {}
    features = {}
    time = pd.read_excel('pfs.xlsx')
    print('feat_names:',feat_names)
    mean_1 = X2[feat_names].mean().values.flatten().tolist()
    mean_2 = X2[feat_names].mean().values.flatten().tolist()
    feat_names.append('Patient_number')
    global pr
    with torch.no_grad():

        print(X1[feat_names].head().to_string())
        for name, input in data_loader:

            name = name[0]

            names.append(name)

            input = input.to(device)
            output, feat = model(input)

            features[name] = feat


            try:
                labels.append(csv_file.loc[csv_file['id'] == name].iloc[0, 1])
                data.append(feat.squeeze().float().to('cpu').numpy())
                outputs.append(output.squeeze().float().to('cpu').numpy())


                row = df[df['id'] == name]
                if not row.empty:
                    # 提取除 'id' 之外的列的数据，并转化为列表
                    values = row.drop(columns=['id']).values.flatten().tolist()
                    clinic_feat.append(values)

                else:
                    clinic_feat.append(clinic_mean_values.tolist())

            except:
                continue
            if name in X1['Patient_number'].values:
                data1.append(X1[feat_names][X1['Patient_number'] == name].drop(columns=['Patient_number']).values.flatten().tolist())
            else:
                data1.append(mean_1)
            if name in X2['Patient_number'].values:
                data2.append(X2[feat_names][X2['Patient_number'] == name].drop(columns=['Patient_number']).values.flatten().tolist())
            else:
                data2.append(mean_2)
            # print(len(X1[feat_names].drop(columns=['Patient_number']).values.flatten().tolist()))
            if name in pr['id'].values:
                # 如果PatientID存在，则更新对应的特征列
                pr.loc[pr['id'] == name, i] = output.squeeze().float().cpu().numpy()
            else:
                # 如果PatientID不存在，则添加新行
                new_row = pd.DataFrame({'id': [name], 'Score': [output.squeeze().float().cpu().numpy()]})
                pr = pd.concat([pr, new_row], ignore_index=True)
    feat_names.remove('Patient_number')
    # df = df.merge(pr, on='id', how='left')
    # df = df.merge(csv_file[['id', 'PFS', 'Status']], on='id', how='left')
    # df = df.drop(columns=['id'])
    # draw_nomogram(df, 'PFS','Status')
    # sys.exit()
    # # 计算最小值和最大值
    # min_val = min(values)
    # max_val = max(values)
    #
    # # 进行最大最小归一化并更新字典
    # results = {key: (value - min_val) / (max_val - min_val)  for key, value in results.items()}
    # print(results)
    # csv_file['pdl1'] = csv_file['id'].map(results)
    # correlation, p_value = pearsonr(csv_file['pdl1'], csv_file['PFS'])
    #
    # print(f"皮尔逊相关系数: {correlation}")
    # print(f"p值: {p_value}")

    # 检查p值是否小于显著性水平 (例如 0.05)
    # if p_value < 0.05:
    #     print("两列变量之间的相关性显著")
    # else:
    #     print("两列变量之间的相关性不显著")
    # time['pdl1'] = time['id'].map(results)
    # time = time.dropna()
    print(feat_names)
    print(len(data1))
    print(len(data2))
    # line(time)
    count = Counter(labels)
    print(count)
    data_array = np.array(data)
    # 计算均值和标准差
    mean = data_array.mean(axis=0)
    std = data_array.std(axis=0)
    # 避免除以零的情况
    std[std == 0] = 1
    # 执行标准化
    normalized_data = (data_array - mean) / std

    outputs_array = np.array(outputs)
    timetime = np.array(timetime)
    # 组合特征和标签到一个大数组中，标签作为最后一列
    # combined_array = np.column_stack((normalized_data, outputs_array))
    combined_array = normalized_data
    # rad_array = np.array(rad_feat)
    # combined_array = np.column_stack((combined_array, rad_array))
    clinic_array = np.array(clinic_feat)
    combined_array = np.column_stack((combined_array, clinic_array))
    labels_array = np.array(labels)
    # print(combined_array[0].shape)
    # combined_array = np.column_stack((combined_array, timetime))
    feature_names = process_strings(feature_names)
    feat_names = process_strings(feat_names)
    # feature_names.extend(["pre-" + item for item in feat_names])
    # feature_names.extend(["post-" + item for item in feat_names])
    return combined_array, labels_array, feature_names, names, data1, data2
def train(X, y, data1, data2, names, fig=None):
    df = pd.DataFrame()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    x1_tensor = torch.tensor(data1, dtype=torch.float32)
    x2_tensor = torch.tensor(data2, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, x1_tensor, x2_tensor, y_tensor)
    dataset = list(zip(names, dataset))
    kf = KFold(n_splits=5, shuffle=True, random_state=46)
    results = [[],[],[],[], []]
    print('输入维度：',X_tensor.shape[1])

    # 记录损失值
    train_losses_per_fold = []
    val_losses_per_fold = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_tensor)):
        hr = 0
        pfs = pd.read_excel('pfs.xlsx')
        aucc = 0
        lossv = 9999
        # 创建数据子集
        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128)

        # 初始化模型、损失函数和优化器
        model = NG(X_tensor.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)

        # 初始化保存AUC的列表
        train_auc_scores = []
        val_auc_scores = []

        val_acc_scores = []
        val_f1_scores = []
        j = []
        # 初始化保存损失的列表
        train_losses = []
        val_losses = []

        num_epochs = 80
        index = 0
        for epoch in range(num_epochs):
            jg = {}
            model.train()
            all_train_labels = []
            all_train_preds = []
            running_train_loss = 0.0


            for name, (inputs, x1, x2, labels) in train_loader:
                x1, x2 = x1.to(device), x2.to(device)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                concatenated_inputs = torch.cat((inputs, x1, x2), dim=1)
                # outputs = model(concatenated_inputs, epoch)
                outputs = model(inputs[:,:], epoch)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

                # 保存预测和标签用于AUC计算
                all_train_labels.extend(labels.tolist())
                all_train_preds.extend(torch.sigmoid(outputs.cpu()).detach().numpy().tolist())

            train_losses.append(running_train_loss / len(train_loader))

            # 计算训练集AUC


            # 验证模式
            model.eval()
            all_val_labels = []
            all_val_preds = []
            running_val_loss = 0.0
            with torch.no_grad():
                for name, (inputs,x1, x2, labels) in val_loader:
                    x1, x2 = x1.to(device), x2.to(device)
                    inputs, labels = inputs.to(device), labels.to(device)
                    concatenated_inputs = torch.cat((inputs, x1, x2), dim=1)
                    # outputs = model(concatenated_inputs)
                    outputs = model(inputs[:, :])
                    loss = criterion(outputs, labels.unsqueeze(1)).item()
                    running_val_loss += loss
                    all_val_labels.extend(labels.tolist())
                    all_val_preds.extend(torch.sigmoid(outputs.cpu()).detach().numpy().tolist())
                    for n, prediction in zip(name, outputs):
                        jg[n] = prediction.item()
                all_train_labels = []
                all_train_preds = []
                for name, (inputs, x1, x2, labels) in train_loader:
                    x1, x2 = x1.to(device), x2.to(device)

                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    concatenated_inputs = torch.cat((inputs, x1, x2), dim=1)
                    # outputs = model(concatenated_inputs)
                    outputs = model(inputs[:, :])
                    # 保存预测和标签用于AUC计算
                    all_train_labels.extend(labels.tolist())
                    all_train_preds.extend(torch.sigmoid(outputs.cpu()).detach().numpy().tolist())
                train_auc = roc_auc_score(all_train_labels, all_train_preds)
                train_auc_scores.append(train_auc)
            val_losses.append(running_val_loss / len(val_loader))

            val_auc, acc, f1 = evaluate_performance(np.squeeze(all_val_preds), np.squeeze(all_val_labels))
            val_auc_scores.append(val_auc)
            val_acc_scores.append(acc)
            val_f1_scores.append(f1)
            if loss < lossv:
                torch.save(model, 'model.pth')
                lossv = loss
                aucc = val_auc
                index = epoch
                m = model
                pfs['mean_value'] = pfs['id'].map(jg)
                pfs = pfs.dropna(subset=['mean_value'])
                # c, _ = calculate_c_index_and_hr(pfs)
                # hr = _['Hazard Ratio']

                train_auc = roc_auc_score(all_train_labels, all_train_preds)
                df[f'{fold}_predict'] = np.squeeze(all_val_preds)
                df[f'{fold}_real'] = np.squeeze(all_val_labels)
            # print(f'epoch:{epoch}, train:{train_auc}, val:{val_auc}')
        train_losses_per_fold.append(train_losses)
        val_losses_per_fold.append(val_losses)

        max_index = max(enumerate(val_auc_scores), key=lambda x: x[1])[0]
        max_index = index
        # print(index)
        # print(f'train:{train_auc_scores[max_index]}, val:{val_auc_scores[max_index]}')

        results[0].append(train_auc_scores[max_index])
        results[1].append(val_auc_scores[max_index])
        results[2].append(val_acc_scores[max_index])
        results[3].append(val_f1_scores[max_index])
        results[4].append(hr)
        # if random.randint(0,4) == fold:
        #     break
    row_means = np.mean(results, axis=1)
    # if row_means[1] > 0.899 and row_means[1] < 0.90:
    #     print(df.head())
    #     plot_mean_roc_curve(df, row_means[1])
    #     sys.exit()
    # if results[1][1] > 0.78 and results[1][1] < 0.79:
    #
    #     pfs = pfs[pfs['mean_value'].notna()]
    #     print(pfs.head())
    #     print(calculate_c_index_and_hr(pfs, 'pfs_img'))
    #     sys.exit()
    if fig:
        # 绘制损失曲线
        plt.figure(figsize=(12, 8))
        for i in range(kf.get_n_splits()):
            plt.plot(train_losses_per_fold[i], label=f'Fold {i+1} Train Loss')
            plt.plot(val_losses_per_fold[i], label=f'Fold {i+1} Val Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Fold')
        plt.legend()
        plt.show()
    # for i in range(5):
    #     print(f'train_auc: {results[0][i]:.3f}, val_auc: {results[1][i]:.3f}, val_Acc: {results[2][i]:.3f}, val_F1: {results[3][i]:.3f}')

    return m, row_means, results[1]

if __name__ == '__main__':
    data = Load('feat', './yh.pth')
    data_loader = DataLoader(dataset=data, batch_size=1)
    mean_out = []
    out = []
    shap_sums = []
    best_names = []
    for i in range(1):
        X, y, feature_names, names, data1, data2 = get_data(data_loader, f'save/best_0.pth', 'yh.xlsx', i,
                                                            '免疫计算用.xlsx')
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        for i in range(10):
            model, auc, j = train(X, y, data1, data2, names)
            model.cpu()
            out.extend(j)
            mean_out.append(auc)
            print(i, ':', np.mean(mean_out, axis=0))

            X_tensor = torch.tensor(X, dtype=torch.float32)
            x1_tensor = torch.tensor(data1, dtype=torch.float32)
            x2_tensor = torch.tensor(data2, dtype=torch.float32)
            # X_tensor = torch.cat((X_tensor, x1_tensor, x2_tensor), dim=1)
            explainer = shap.GradientExplainer(model, X_tensor[:100])  # 使用模型和训练数据的一个子集
            shap_values = explainer.shap_values(X_tensor[:100])

            # 检查 shap_values 是否为列表，如果是，则选择第一个输出
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # 确保 shap_values 的形状是 (n_samples, n_features)
            print(f"SHAP values shape: {shap_values.shape}")

            shap_sums.append(shap_values)

    # 确保 shap_sums 形状一致，并计算均值
    shap_sums = np.array(shap_sums)
    mean_shap_sum = np.mean(shap_sums, axis=0)

    # 检查 mean_shap_sum 的形状
    importance_first_512 = np.sum(np.abs(mean_shap_sum[:, 512]))

    # 计算后面所有特征的总重要性
    importance_rest_sum = np.sum(np.abs(mean_shap_sum[:, 512:]))

    # 创建特征重要性 DataFrame
    mean_shap_importance = np.mean(np.abs(mean_shap_sum), axis=0)
    importance_second_512 = np.sum(mean_shap_importance[:512])
    importance_rest_sum = np.sum(mean_shap_importance[512:512+8])
    importance_rest_sum2 = np.sum(mean_shap_importance[512+8:])
    feature_importance = pd.DataFrame(list(zip(feature_names, mean_shap_importance)),
                                      columns=['Feature', 'SHAP Importance'])
    feature_importance.sort_values(by='SHAP Importance', ascending=False, inplace=True)

    # 打印特征重要性
    print(feature_importance.head(20))

    # 打印计算结果
    print(f"Total importance of the first 512 features: {importance_first_512}")
    print(f"Total importance of features after the 512th{feature_names[512:512+8]}: {importance_rest_sum}")
    print(f"Total importance of features after the 512th2{feature_names[512+8:]}: {importance_rest_sum2}")


    # 绘制SHAP summary plot
    shap.summary_plot(mean_shap_sum, features=X_tensor.numpy(), feature_names=feature_names, show=True,
                      plot_size=(9, 9.5))



