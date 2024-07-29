import argparse
import os
from collections import Counter
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import  DataLoader, TensorDataset, Subset

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import KFold
import numpy as np
import torch.optim as optim

from dataset import pdl1_task_load
from utils import feature_selection_with_lasso, evaluate_performance, get_longitudinal, get_cancer, EarlyStopping
from models import NN, MN, MR, MMDL

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Process some paths for the dataset.')

# 添加变量
parser.add_argument('--task', type=str, default='Longitudinal', choices=['Immunotherapy', 'PFS', 'Longitudinal'], help='The task to perform')
parser.add_argument('--files_path', type=str, default='feat_Immunotherapy', help='Path to the files')
parser.add_argument('--coords_path', type=str, default='yh.pth', help='Path to the pixel sizes for each SVS file')
parser.add_argument('--cli_and_rad_path', type=str, default='dataset/feature.xlsx', help='Path to the combined clinical and radiomic variables file')
parser.add_argument('--combined_path', type=str, default='dataset/combined.xlsx', help='Path to the combined clinical and radiomic variables file')
parser.add_argument('--radiomics_before_path', type=str, default='dataset/Immunotherapy_before_feature.xlsx', help='Path to the radiomic variables file')
parser.add_argument('--radiomics_after_path', type=str, default='dataset/Immunotherapy_after_feature.xlsx', help='Path to the radiomic variables file')
parser.add_argument('--combined_path', type=str, default='dataset/combined.xlsx', help='Path to the combined clinical and radiomic variables file')


parser.add_argument('--label_path', type=str, default='Immunotherapy.xlsx', help='Path to the label file')
parser.add_argument('--save', type=str, default='save', help='Path to save the output')
parser.add_argument('--model_path', type=str, default='save/PD-L1_0.pth', help='Path to save the output')

args = parser.parse_args()
task = args.task
files_paths = args.files_path
coords_paths = args.coords_path
cli_and_rad_paths = args.cli_and_rad_path
before_path = args.radiomics_before_path
after_path = args.radiomics_after_path
label_path = args.label_path
model_path = args.model_path
save_path = args.save
combined_path = args.cli_and_rad_path
def get_data(dataset, model_path, csv_path, combined_path=None):
    input_dim = dataset.get_dim()
    scaler = StandardScaler()
    data_before, data_after, feat_names = get_longitudinal(before_path, after_path)
    X1 = data_before
    X2 = data_after
    scaler = StandardScaler()
    def standardize_features():
        feat_names.remove('id')
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
        feat_names.append('id')
        return X1, X2
    X1, X2 = standardize_features()
    data1 = []
    data2 = []

    csv_file = pd.read_excel(csv_path)
    df = pd.read_excel(combined_path)

    names = []
    data = []
    labels = []
    combined_feat = []

    df = feature_selection_with_lasso(df, csv_file, 0.05)

    model = MMDL(input_dim).to(device)
    static_dict = torch.load(model_path)
    model.load_state_dict(static_dict)
    model.eval()
    features = {}
    mean_1 = X2[feat_names].mean().values.flatten().tolist()
    mean_2 = X2[feat_names].mean().values.flatten().tolist()


    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        for name, input in data_loader:
            name = name[0]
            names.append(name)
            input = input.to(device)
            output, feat = model(input)
            features[name] = feat


            labels.append(csv_file.loc[csv_file['id'] == name].iloc[0, 1])
            data.append(feat.squeeze().float().to('cpu').numpy())

            row = df[df['id'] == name]
            combined_feat.append(row.drop('id', axis=1).values)
            data1.append(X1[feat_names][X1['id'] == name].drop(columns=['id']).values.flatten().tolist())
            data2.append(X2[feat_names][X2['id'] == name].drop(columns=['id']).values.flatten().tolist())

    feat_names.remove('id')

    count = Counter(labels)
    print(count)
    data_array = np.array(data)

    mean = data_array.mean(axis=0)
    std = data_array.std(axis=0)
    # 避免除以零的情况
    std[std == 0] = 1
    # 执行标准化
    normalized_data = (data_array - mean) / std

    combined_array = normalized_data
    clinic_array = np.squeeze(np.array(combined_feat))

    combined_array = np.column_stack((combined_array, clinic_array))
    labels_array = np.array(labels)

    return combined_array, labels_array, names, data1, data2
def train(X, y, data1, data2, names, save_path, task, fig=None):
    df = pd.DataFrame()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    x1_tensor = torch.tensor(data1, dtype=torch.float32)
    x2_tensor = torch.tensor(data2, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, x1_tensor, x2_tensor, y_tensor)
    dataset = list(zip(names, dataset))
    kf = KFold(n_splits=5, shuffle=True, random_state=46)
    results = [[],[],[],[]]
    print('输入维度：',X_tensor.shape[1])

    # 记录损失值
    train_losses_per_fold = []
    val_losses_per_fold = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_tensor)):
        # 创建数据子集
        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128)

        # 初始化模型、损失函数和优化器
        if task == 'Immunotherapy' or task == 'PFS':
            model = MN(X_tensor.shape[1]).to(device)
        else:
            model = MR(X_tensor.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)

        # 初始化保存AUC的列表
        train_auc_scores = []

        val_auc_scores = []
        val_acc_scores = []
        val_f1_scores = []
        # 初始化保存损失的列表
        train_losses = []
        val_losses = []
        num_epochs = 80
        index = 0
        early_stopping = EarlyStopping(patience=15, stop_epoch=15, verbose=False)
        for epoch in range(num_epochs):
            model.train()
            all_train_labels = []
            all_train_preds = []
            running_train_loss = 0.0


            for name, (inputs, x1, x2, labels) in train_loader:
                x1, x2 = x1.to(device), x2.to(device)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                if task == 'Longitudinal':
                    concatenated_inputs = torch.cat((inputs, x1, x2), dim=1)
                    outputs = model(concatenated_inputs, epoch)
                else:
                    outputs = model(inputs[:,:], epoch)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

                # 保存预测和标签用于AUC计算
                all_train_labels.extend(labels.tolist())
                all_train_preds.extend(torch.sigmoid(outputs.cpu()).detach().numpy().tolist())
            train_auc = roc_auc_score(all_train_labels, all_train_preds)
            train_auc_scores.append(train_auc)
            train_losses.append(running_train_loss / len(train_loader))

            # 验证模式
            model.eval()
            all_val_labels = []
            all_val_preds = []
            running_val_loss = 0.0
            with torch.no_grad():
                for name, (inputs,x1, x2, labels) in val_loader:
                    x1, x2 = x1.to(device), x2.to(device)
                    inputs, labels = inputs.to(device), labels.to(device)
                    if task == 'Longitudinal':
                        concatenated_inputs = torch.cat((inputs, x1, x2), dim=1)
                        outputs = model(concatenated_inputs, epoch)
                    else:
                        outputs = model(inputs[:, :], epoch)
                    loss = criterion(outputs, labels.unsqueeze(1)).item()
                    running_val_loss += loss
                    all_val_labels.extend(labels.tolist())
                    all_val_preds.extend(torch.sigmoid(outputs.cpu()).detach().numpy().tolist())

            val_losses.append(running_val_loss / len(val_loader))

            val_auc, acc, f1 = evaluate_performance(np.squeeze(all_val_preds), np.squeeze(all_val_labels))
            val_auc_scores.append(val_auc)
            val_acc_scores.append(acc)
            val_f1_scores.append(f1)
            early_stopping(epoch, running_val_loss, model, optimizer,
                           ckpt_name=os.path.join(save_path,f'best_model.pth'))
            if early_stopping.early_stop:
                print("stop")
                index = epoch
                break


        results[0].append(train_auc_scores[index])
        results[1].append(val_auc_scores[index])
        results[2].append(val_acc_scores[index])
        results[3].append(val_f1_scores[index])

    row_means = np.mean(results, axis=1)

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

    return  row_means, results[1]

if __name__ == '__main__':

    data = pdl1_task_load(files_paths, coords_paths, cli_and_rad_paths)

    X, y, names, data1, data2 = get_data(data, model_path, label_path, combined_path)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    auc, j = train(X, y, data1, data2, names, save_path, task)




