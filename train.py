import argparse
import warnings
from models import MMDL
warnings.filterwarnings("ignore")
import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import  KFold
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from models import MMDL, NN
from utils import EarlyStopping, get_cancer
from dataset import pdl1_task_load, risk_task_load
batch_size = 1

csv_path = r'./PD-L1.csv'
save_path = 'save'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Process some paths for the dataset.')

# 添加变量
parser.add_argument('--task', type=str, default='PD-L1', choices=['PD-L1', 'risk'], help='The task to perform')
parser.add_argument('--files_path', type=str, default='feat_PD-L1', help='Path to the files')
parser.add_argument('--coords_path', type=str, default='coord.pt', help='Path to the pixel sizes for each SVS file')
parser.add_argument('--cli_and_rad_path', type=str, default='dataset/features.xlsx', help='Path to the combined clinical and radiomic variables file')
parser.add_argument('--label_path', type=str, default='PD-L1.xlsx', help='Path to the label file')
parser.add_argument('--save', type=str, default='save', help='Path to save the output')



def val(val_loader, model):

    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = 0

    predict = []
    real = []
    out = []
    model.eval()
    with torch.no_grad():
        for name, inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output, _ = model(inputs)
            loss = criterion(output, labels)
            output = nn.Softmax(dim=1)(output)[:,1].squeeze().to('cpu').detach().numpy()
            labels = labels.squeeze().to('cpu').detach().numpy()
            predict.append(output)
            real.append(labels)
            out.append(output)
            val_loss += loss.item()


    auc = roc_auc_score(real, out)
    print('\033[92mAUC:', auc, '\033[0m\tval_loss:', val_loss / len(val_loader))

    return auc, val_loss / len(val_loader)


def train(data_loader, val_loader, n, input_dim):

    model = MMDL(input_dim).to(device)

    weight = torch.tensor([1, 1]).float()
    loss_fn = nn.CrossEntropyLoss(weight=weight).to(device)

    loss_f1 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # 进行训练
    num_epochs = 50
    best_cor = 0
    model.train()

    val_Loss = []
    loss = []
    xx = []
    early_stopping = EarlyStopping(patience=15, stop_epoch=15, verbose=False)
    for i in range(num_epochs):

        xx.append(i)
        predict = []
        real = []
        out = []
        train_total_loss = 0
        l1_total_loss = 0

        for name, x, y in data_loader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output, _ = model(x)

            # 计算损失
            train_loss = loss_fn(output, y)
            loss_1 = loss_f1(output, y)


            output = nn.Softmax(dim=1)(output)[:,1].squeeze().to('cpu').detach().numpy()
            y = y.to('cpu').detach().numpy()
            predict.append(output)
            out.append(output)
            real.append(y)

            train_loss.backward(retain_graph=True)

            optimizer.step()
            train_total_loss += train_loss.item()
            l1_total_loss += loss_1.item()

        auc = roc_auc_score(real, out)

        loss.append(train_total_loss / len(data))

        print(f'{i}/{num_epochs}' ,'AUC:', auc, '\ttrain_loss:', train_total_loss / len(data))

        val_auc, val_loss = val(val_loader, model)
        val_Loss.append(val_loss)

        early_stopping(i, -val_auc, model, optimizer,
                       ckpt_name=os.path.join(save_path,f'best_model_{n}.pth'))


        if early_stopping.early_stop:
            print("stop")
            break


def main(data):
    input_dim = data.get_dim()
    kf = KFold(n_splits=5, shuffle=True, random_state=46)
    for fold, (train_index, val_index) in enumerate(kf.split(data)):

        # 获取训练和验证数据的索引
        train_fold = torch.utils.data.dataset.Subset(data, train_index)
        val_fold = torch.utils.data.dataset.Subset(data, val_index)
        train_loader = DataLoader(dataset=train_fold, batch_size=1, shuffle=True)
        val_loader = DataLoader(dataset=val_fold, batch_size=1, shuffle=True)
        train(train_loader, val_loader, fold, input_dim)




if __name__ == '__main__':
    args = parser.parse_args()

    files_paths = args.files_path
    coords_paths = args.coords_path
    cli_and_rad_paths = args.cli_and_rad_path
    label_path = args.label_path
    if args.task == 'PD-L1':
        data = pdl1_task_load(files_paths, coords_paths, cli_and_rad_paths, label_path)
    else:
        data = risk_task_load(files_paths, coords_paths, cli_and_rad_paths, label_path)
    main(data)
