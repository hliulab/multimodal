import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import get_cancer, feature_selection_with_lasso
from models import NN


class risk_task_load(Dataset):

    # - files_path: This variable represents the file path where the files are stored.
    # - coords_path: This variable indicates the pixel size for each SVS (Scalable Vector Graphics) file.
    # - cli_and_rad_path: This variable contains the path to the combined dataframe of clinical and radiomic variables.
    # - label_path: This variable represents the path to the label file.

    def __init__(self, files_path, coords_path, cli_and_rad_path, label_path):
        print('文件路径：', files_path)
        self.names = []
        self.data = {}
        self.csv_file = pd.read_excel(label_path)
        combined_df = pd.read_excel(cli_and_rad_path)

        combined_df = feature_selection_with_lasso(combined_df, self.csv_file)
        coords = torch.load(coords_path, map_location='cpu')
        print(len(coords))

        for i in tqdm(os.listdir(files_path)):
            name = i[:-3]

            l, r = coords[name]

            q = torch.load(os.path.join(files_path, i), map_location='cpu')
            features = []
            coordinates = []
            for (x, y), tensor in q.items():
                features.append(tensor)
                coordinates.append([int(x) / int(l) , int(y) / int(r) ])

            features_tensor = torch.stack(features).squeeze()
            # Standardization
            mean = features_tensor.mean(dim=1, keepdim=True)
            std = features_tensor.std(dim=1, keepdim=True)
            features_tensor = (features_tensor-mean)/std

            coordinates_tensor = torch.tensor(coordinates)
            combined_tensor = torch.cat((features_tensor, coordinates_tensor), dim=1)


            indices = torch.randperm(combined_tensor.shape[0])[:]
            selected_tensor = combined_tensor[indices]
            sort_key = selected_tensor[:, -2] + selected_tensor[:, -1] / 1000000
            # 根据组合键排序
            sorted_indices = torch.argsort(sort_key)
            sorted_tensor = selected_tensor[sorted_indices]

            row = combined_df[combined_df['id'] == name]
            row_tensor = torch.tensor(row.drop('id', axis=1).values)
            combined_tensor = torch.cat((sorted_tensor, row_tensor.repeat(sorted_tensor.shape[0], 1)), dim=1)


            self.names.append(name)
            self.data[name] = combined_tensor[:].to(torch.float32)

        print(len(self.names))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item) -> tuple:

        name = self.names[item]
        label = self.csv_file.loc[self.csv_file['id'] == name].iloc[0, 1]
        return name, self.data[name], label

class pdl1_task_load(Dataset):

    def __init__(self, files_path, coords_path, cli_and_rad_path, label_path):

        print('file path：', files_path)
        self.names = []
        self.data = {}
        self.csv_file = pd.read_excel(label_path)
        df = pd.read_excel(cli_and_rad_path)
        df = feature_selection_with_lasso(df, self.csv_file, 0.01)

        coords = torch.load(coords_path, map_location='cpu')
        for i in tqdm(os.listdir(files_path)):
            name = i[:-3]
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

            features_tensor = torch.stack(features).squeeze()

            mean = features_tensor.mean(dim=1, keepdim=True)
            std = features_tensor.std(dim=1, keepdim=True)
            features_tensor = (features_tensor-mean)/std

            coordinates_tensor = torch.tensor(coordinates)
            row = df[df['id'] == name]
            row_tensor = torch.tensor(row.drop('id', axis=1).values)
            combined_tensor = torch.cat((features_tensor, coordinates_tensor), dim=1)
            row_tensor = row_tensor.repeat(len(features), 1)

            combined_tensor = torch.cat((combined_tensor, row_tensor), dim=1)


            self.names.append(name)
            self.data[name] = combined_tensor[:].to(torch.float32)

        print(len(self.names))


    def __len__(self):
        return len(self.names)

    def __getitem__(self, item) -> tuple:
        name = self.names[item]
        label = self.csv_file.loc[self.csv_file['id'] == name].iloc[0, 1]
        return name, self.data[name][:, :], label

    def get_dim(self):
        return self.data[self.names[0]].shape[1]

class pdl1_task_load(Dataset):

    def __init__(self, files_path, coords_path, cli_and_rad_path):

        print('file path：', files_path)
        self.names = []
        self.data = {}
        df = pd.read_excel(cli_and_rad_path)
        coords = torch.load(coords_path, map_location='cpu')
        for i in tqdm(os.listdir(files_path)):
            name = i[:-3]
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

            features_tensor = torch.stack(features).squeeze()

            mean = features_tensor.mean(dim=1, keepdim=True)
            std = features_tensor.std(dim=1, keepdim=True)
            features_tensor = (features_tensor-mean)/std

            coordinates_tensor = torch.tensor(coordinates)

            row = df[df['id'] == name]
            row_tensor = torch.tensor(row.drop('id', axis=1).values)
            combined_tensor = torch.cat((features_tensor, coordinates_tensor), dim=1)
            row_tensor = row_tensor.repeat(len(features), 1)

            combined_tensor = torch.cat((combined_tensor, row_tensor), dim=1)


            self.names.append(name)
            self.data[name] = combined_tensor[:].to(torch.float32)

        print(len(self.names))


    def __len__(self):
        return len(self.names)

    def __getitem__(self, item) -> tuple:
        name = self.names[item]
        return name, self.data[name][:, :]

    def get_dim(self):
        return self.data[self.names[0]].shape[1]

# data = pdl1_task_load('feat_Immunotherapy', 'yh.pth', 'feature.xlsx')