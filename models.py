import torch
from torch import nn


class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    def forward(self, x):
        x = self.fc(x)
        return x

class MMDL(nn.Module):
    def __init__(self, input_dim, transformer_dim=512, num_heads=8, num_layers=2, hidden_dim=2048, output_dim=2):
        super(MMDL, self).__init__()

        class Permute(nn.Module):
            def __init__(self, *dims):
                super(Permute, self).__init__()
                self.dims = dims

            def forward(self, x):
                return x.permute(self.dims)

        encoder_layers = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads,
                                                    dim_feedforward=hidden_dim)
        self.features = nn.Sequential(
            nn.Linear(input_dim, transformer_dim),
            Permute(1, 0, 2),
            nn.TransformerEncoder(encoder_layers, num_layers=num_layers),
        )
        self.fc = nn.Linear(transformer_dim, output_dim)
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

class MN(nn.Module):
    def __init__(self, input_size):
        super(MN, self).__init__()
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
class MR(nn.Module):
    def __init__(self, input_size):
        super(MR, self).__init__()
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