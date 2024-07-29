import random
import sys
import warnings

from statistics import mean

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import gc
import os
from collections import OrderedDict
from sklearn.metrics import cohen_kappa_score
import numpy
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageFile
import time
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from utils import feature_selection_with_lasso
import torch.nn.functional as F
from sklearn.metrics import explained_variance_score, r2_score, classification_report, mean_squared_error, \
    mean_absolute_error, roc_auc_score

from CLAM.utils.early_stopping_utils import EarlyStopping
from GRU import data_before, data_after, feature_names

data_analyze = {}
batch_size = 1
#

# data_path = './all_pic_256'
# val_data_path = './val'
results = {}
csv_path = r'./PD-L1.csv'
save_path = 'save'
neg_analyze = {}
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        # 定义一个全连接层
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        # 应用sigmoid激活函数
        x = self.fc(x)
        # x = torch.sigmoid(self.fc(x))
        return x

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
            nn.TransformerEncoder(encoder_layers, num_layers=num_layers),
        )
        self.fc = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = x.mean(dim=0)
        feat = x
        # x = torch.sigmoid(x)
        x = self.fc(x)
        return x, feat
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

    def __init__(self, csv_path, files_path, coords_path):
        '''*****************'''
        self.X1 = data_before
        self.X2 = data_after
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.X1, self.X2 = self.standardize_features()
        '''*****************'''
        print('文件路径：', files_path)
        self.names = []
        self.data = {}
        self.n = []
        coords = torch.load(coords_path, map_location='cpu')
        rad_feat = 'pyradiomics_feature.csv'
        df = pd.read_csv(rad_feat)

         # Standardize each column except for the ID column
        for col in df.columns:
            if col != 'id':  # Assuming 'id' is the name of your ID column
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        # clinic = feature_selection_with_lasso(pd.read_excel('pdgai.xlsx'), self.csv_file[['id', 'PD-L1_50']], 0.01)
        # df = pd.merge(df, clinic, on='id')
        mean_tensor  = torch.tensor(df.drop(columns=['id'], axis=1).mean().values)
        for i in tqdm(os.listdir(files_path)):
            name = i[:-3]
            results[name] = []
            l, r = coords[name]

            q = torch.load(os.path.join(files_path, i), map_location='cpu')
            features = []
            coordinates = []
            for (x, y), tensor in q.items():
                features.append(tensor)
                coordinates.append([int(x) / int(l) , int(y) / int(r) ])
            sorted_index = get_cancer(features)
            features = [features[index] for index in sorted_index][:len(features)//3 * 2]
            coordinates = [coordinates[index] for index in sorted_index][:len(coordinates)//3 * 2]

            # features = torch.tensor([item.cpu().detach().numpy() for item in features])
            # if len(features) < 2000:
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
                # continue
                row_tensor = mean_tensor

            combined_tensor = torch.cat((features_tensor, coordinates_tensor), dim=1)
            n = combined_tensor.shape[0]
            row_tensor = row_tensor.repeat(n, 1)


            # print(combined_tensor.shape)
            # indices = torch.randperm(n)[:]
            # selected_tensor = combined_tensor[indices]
            # sort_key = selected_tensor[:, -2] + selected_tensor[:, -1] / 1000000
            # # 根据组合键排序
            # sorted_indices = torch.argsort(sort_key)
            # sorted_tensor = selected_tensor[sorted_indices]
            # combined_tensor = torch.cat((sorted_tensor, row_tensor), dim=1)
            combined_tensor = torch.cat((combined_tensor, row_tensor), dim=1)

            # print(n)


            # print(combined_tensor.shape)
            self.names.append(name)

            self.data[name] = combined_tensor[:].to(torch.float32)

            # print(sorted_tensor.shape)

            neg_analyze[name] = 0


            # print(self.data[name].shape).385


        # column_name = 'PD-L1'  # 或者直接使用列名，例如 'Column2'
        #
        # # 最大最小归一化
        # max_value = self.csv_file[column_name].max()
        # min_value = self.csv_file[column_name].min()
        # self.csv_file[column_name] = (self.csv_file[column_name] - min_value) / (max_value - min_value)
        # self.csv_file.iloc[:, 1] = self.csv_file.iloc[:, 1].apply(lambda x: 1 if x == 'Positive' else 0)
        # csv_ids = set(self.csv_file['id'].astype(str).str[:12])
        #
        # # 提取self.names中每个元素的前12个字符，并转化为集合
        # names_ids = set(item[:12] for item in self.names)
        #
        # # 找到在csv_ids中存在但在names_ids中不存在的ID
        # missing_ids = csv_ids - names_ids

        # 打印这些ID
        # for id in missing_ids:
        #     print(id)
        self.csv_file = pd.read_csv(csv_path)
        self.names = [item for item in self.names if item[:12] in self.csv_file['id'].values]
        _0 = 0
        _1 = 0
        # self.csv_file[:, 1] = self.csv_file[:, 1].apply(lambda x: 1 if x > 10 else 0)
        for name in self.names:
            l = self.csv_file.loc[self.csv_file['id'] == name[:12]].iloc[0, 1]
            if l <= 10:
                _0 += 1
            else:
                _1 += 1
        print(_0, '  ',_1)
        print(len(self.names))


    def __len__(self):
        return len(self.names)

    def __getitem__(self, item) -> tuple:
        name = self.names[item]
        label = self.csv_file.loc[self.csv_file['id'] == name[:12]].iloc[0, 1]
        if label > 10:
            label = 1
        else:
            label = 0
        return name, self.data[name][:, :], label

    def get_X2_by_filename(self, filename):
        for index, row in self.X2.iterrows():
            csv_name_prefix = row.loc["Patient_number"]
            if filename == csv_name_prefix:
                return row
        return None
    def get_X1_by_filename(self, filename):
        for index, row in self.X1.iterrows():
            csv_name_prefix = row.loc["Patient_number"]
            if filename == csv_name_prefix:
                return row
        return None

    def standardize_features(self):
        # Check if feature names are in both dataframes
        for feature in self.feature_names:
            if feature not in self.X1.columns or feature not in self.X2.columns:
                raise ValueError(f"Feature '{feature}' is not present in both datasets")

        # Extract only the columns to be standardized
        features_before = self.X1[self.feature_names]
        features_after = self.X2[self.feature_names]

        # Standardize the features
        self.scaler.fit(features_before)
        standardized_features_before = self.scaler.transform(features_before)
        standardized_features_after = self.scaler.transform(features_after)

        # Replace the columns in the original dataframes with the standardized values
        self.X1[self.feature_names] = standardized_features_before
        self.X2[self.feature_names] = standardized_features_after

        return self.X1, self.X2
def val(val_loader, best_cor, model):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 1024 * 2  # 输入大小为两个数组的拼接
    # model = TT().to(device)
    #
    # static_dict = torch.load('./logistic_tem.pth')
    # model.load_state_dict(static_dict)
    # criterion = nn.MSELoss().to(device)
    weight = torch.tensor([1, 1]).float()
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)

    val_loss = 0
    # total = 0
    # correct = 0

    predict = []
    real = []
    out = []
    # model.eval()
    sum = 0
    with torch.no_grad():
        for name, inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # x1, x2 = x1.to(device), x2.to(device)
            # print(inputs.shape)

            output, _ = model(inputs)
            # labels = labels.unsqueeze(1).float()
            loss = criterion(output, labels)
            # print(outputs)
            # print(labels.view(-1, 1))
            output = nn.Softmax(dim=1)(output)[:,1].squeeze().to('cpu').detach().numpy()
            labels = labels.squeeze().to('cpu').detach().numpy()
            predict.append(output)
            real.append(labels)
            out.append(output)
            #
            # output = [1 if output >= 0.50 else 0]
            # labels = [1 if labels >= 0.50 else 0]


            # if output == labels:
            #     sum += 1
            val_loss += loss.item()

            # probabilities = torch.sigmoid(outputs)
            # predictions = (probabilities > _).float()
            #
            # total += labels.size(0)
            # correct += (predictions == labels.view(-1, 1)).sum().item()
            # for i in range(batch_size):
            #     e[name[i]] = predictions[i]
    # print(f"Val Loss:{val_loss / len(val_loader)}\n"
    #       f"Acc:{correct / total}"
    #       )
    # auc = np.corrcoef(real, predict)[0][1]
    # real = [0 if sample < 0.5 else 1 for sample in real]
    auc = roc_auc_score(real, out)
    print('\033[92mAUC:', auc, '\033[0m\tval_loss:', val_loss / len(val_loader))
    # print(
    #     f'\033[92m{classification_report(np.where(np.array(real) < 0.5, 0, 1), np.where(np.array(predict) < 0.5, 0, 1))}\033[0m')
    # print(f'\033[92m{classification_report(np.where(np.array(real) <= 0.5, 0, 1), np.where(np.array(predict) <= 0.5, 0, 1))}\033[0m')

    # compute_data(val_loader, 1)
    # correct = np.corrcoef(predict, real)[0][1]
    # if best_cor < correct:
    #     torch.save(model.state_dict(), './logistic_best.pth')

    return auc, val_loss / len(val_loader)


def train(data, val_loader, n):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # input_size = 1024 * 2  # 输入大小为两个数组的拼接
    input_dim = 1024
    transformer_dim = 512
    num_heads = 8
    num_layers = 2
    hidden_dim = 2048
    output_dim = 1  # 假设我们预测一个标量值
    model = TT(input_dim, transformer_dim, num_heads, num_layers, hidden_dim, output_dim).to(device)

    # model = CLAM_MB_Reg().to(device)

    # 定义损失函数和优化器
    # criterion = nn.MSELoss().to(device)
    # loss_l1 = nn.Loss
    # loss_fn = nn.HuberLoss()

    # loss_fn = nn.MSELoss()
    weight = torch.tensor([1, 1]).float()
    loss_fn = nn.CrossEntropyLoss(weight=weight).to(device)


    # loss_fn = nn.L1Loss()
    loss_f1 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # 进行训练
    num_epochs = 50
    best_cor = 0
    model.train()

    # for name, param in logistic_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, ':', param.size())
    val_Loss = []
    loss = []
    xx = []
    early_stopping = EarlyStopping(patience=15, stop_epoch=15, verbose=False)
    for i in range(num_epochs):

        # for x, y in tqdm(data):
        xx.append(i)
        predict = []
        real = []
        out = []
        train_total_loss = 0
        l1_total_loss = 0
        sum = 0
        # for name, x, y in tqdm(data, desc=f'train epoch:{i}/{num_epochs},  第{n}折'):
        for name, x, y in data:
            x, y = x.to(device), y.to(device)
            # x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            # print(x.shape)

            # 将拼接后的数组输入到逻辑回归模型进行预测

            # output = output.to(device)
            y = y#.unsqueeze(1).float()
            output, _ = model(x)

            # 计算损失
            train_loss = loss_fn(output, y)
            loss_1 = loss_f1(output, y)
            # loss_2 = loss_l1(output, y)

            # output = output.squeeze().to('cpu').detach().numpy()
            # y = y.squeeze().to('cpu').detach().numpy()

            output = nn.Softmax(dim=1)(output)[:,1].squeeze().to('cpu').detach().numpy()
            y = y.to('cpu').detach().numpy()
            predict.append(output)
            out.append(output)
            real.append(y)

            # output = [1 if output >= 0.5 else 0]
            # y = [1 if y >= 0.5 else 0]
            # output = [1 if output > 0.5 else 0]
            # y = [1 if y > 0.5 else 0]
            # if output == y:
            #     sum += 1

            # 反向传播和优化
            # print("Model training state before backward: ", model.training)  # 打印训练状态
            train_loss.backward(retain_graph=True)
            # loss_1.backward()
            optimizer.step()
            train_total_loss += train_loss.item()
            l1_total_loss += loss_1.item()

        # auc = np.corrcoef(real, predict)[0][1]
        # real = [0 if sample < 0.5 else 1 for sample in real]
        auc = roc_auc_score(real, out)
        # auc = r2_score(real, predict)
        # explained = explained_variance_score(real, predict)
        # pl.append(tem[0][1])
        loss.append(train_total_loss / len(data))

        print(f'{i}/{num_epochs}' ,'AUC:', auc, '\ttrain_loss:', train_total_loss / len(data))

        val_auc, val_loss = val(val_loader, best_cor, model)
        # val_loss = train_total_loss
        val_Loss.append(val_loss)

        early_stopping(i, -val_auc, model, optimizer,
                       ckpt_name=os.path.join(save_path,f'PD-L1_{n}.pth'))


        if early_stopping.early_stop:
            print("stop")
            break
    # plt.figure(figsize=(10, 5))
    # plt.plot(xx, loss, 'b', label='Training loss')
    # plt.plot(xx, val_Loss, 'r', label='Training loss')
    # plt.savefig(f'./loss_{n}.png')



Max = 0
index = 0
o = []
def main(data, index):
    mama = 0
    dfdf = pd.DataFrame()
    rr = []
    pp = []
    df = pd.DataFrame()
    i = 0
    # kf = KFold(n_splits=5, shuffle=True, random_state=random)
    plt.figure()
    kf = KFold(n_splits=5, shuffle=True, random_state=46)
    for fold, (train_index, val_index) in enumerate(kf.split(data)):

        # 获取训练和验证数据的索引
        train_fold = torch.utils.data.dataset.Subset(data, train_index)
        val_fold = torch.utils.data.dataset.Subset(data, val_index)
        train_loader = DataLoader(dataset=train_fold, batch_size=1, shuffle=True)
        val_loader = DataLoader(dataset=val_fold, batch_size=1, shuffle=True)
        train(train_loader, val_loader, fold)
        a, r, p = zaishishi(train_loader, val_loader, fold)
        # df[f'{fold}_real'] = r
        # df[f'{fold}_predict'] = p
        i += a
        o.append(a)
        rr.extend(r)
        pp.extend(p)


    # df.to_excel('logistical.xlsx', index=False)
    print(o)
    print(mean(o))

    # print(Max)




def zaishishi(train_loader, val_loader, i):
    input_dim = 1024
    transformer_dim = 512
    num_heads = 8
    num_layers = 2
    hidden_dim = 2048
    output_dim = 1  # 假设我们预测一个标量值
    model = TT(input_dim, transformer_dim, num_heads, num_layers, hidden_dim, output_dim).to(device)
    static_dict = torch.load(os.path.join(save_path, f'./PD-L1_{i}.pth'))
    model.load_state_dict(static_dict)
    model.eval()
    outputs = []
    p = []
    r = []
    with torch.no_grad():

        for name, input, labels in val_loader:
            input, labels = input.to(device), labels.to(device)

            output, _ = model(input)
            output = nn.Softmax(dim=1)(output)[:, 1].squeeze().to('cpu').detach().numpy()
            labels = labels.to('cpu').detach().numpy()

            outputs.append(output)
            results[name[0]].append(labels)
            print(output, '  ', labels)

            p.append(output)
            r.append(labels)
        auc = roc_auc_score(r, p)
        # auc = np.corrcoef(r, p)[0][1]
        print(auc)

        return auc, r, p


def cro_val(train_data, val_data):
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)
    train(train_loader, val_loader, 0)

    o[i].append(zaishishi(train_loader, val_loader, i))
    print(o)
if __name__ == '__main__':

    data = Load(csv_path,'./feat_PD-L1', './coord.pt')



    for i in range(1):

        print(i)
        main(data, i)
    torch.save(results, './1024.pth')
