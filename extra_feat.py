import argparse
import os

from collections import OrderedDict

import h5py
import pandas as pd
import torch

import openslide
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from tqdm import tqdm

from CLAM.models.resnet_custom import resnet50_baseline
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Process some paths for the dataset.')


parser.add_argument('--file_path', type=str, default='tile_results/h5file', help='Path to the input file')
parser.add_argument('--model_path', type=str, default='best.pth.tar', help='Path to the model file')
parser.add_argument('--save_path', type=str, default='feat', help='Path to save the output')


class Resnet50Extractor():
    """
    MoCoExtractor feature extractor (resnet50)
    New key: sample['feature'][key] of shape (Z, 2048, H//32, W//32)

    Parameters
    ----------
        key: str
            name of the model

        device: str
            'cuda' to run on GPU, 'cpu' to run on CPU
    """

    def __init__(self, key, model_path):

        self.key = key

        check_point = torch.load(model_path, map_location=device)
        print(check_point.keys())
        state_dict = check_point['state_dict']
        model = resnet50_baseline()
        new_sd = OrderedDict()
        for k in list(state_dict.keys()):
            # 只要encoder_q
            if k.startswith('module.encoder_q'):
                new_sd[k[len("module.encoder_q."):]] = state_dict[k]
        missing_key = model.load_state_dict(new_sd, strict=False)
        assert set(missing_key.unexpected_keys) == {"fc.0.weight", "fc.0.bias", "fc.2.weight", "fc.2.bias"}

        for name, parameter in model.named_parameters():
            parameter.requires_grad = False
        for name, param in model.named_parameters():
            if 'layer3.4' in name or 'layer3.5' in name:
                param.requires_grad = True

        model.eval()
        model.to(device)

        self.model = model

    def __call__(self, sample):


        features = self.model(sample)


        return features

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(file_path)
        h5py_file = h5py.File(file_path, "r")
        dset = h5py_file['coords']

        self.attr = {}
        for name, value in dset.attrs.items():
            self.attr[name] = value
        self.results = dset[:]
        svs_path = os.path.join('vis', os.path.basename(file_path))[:-3] + '.svs'

        self.wsi = openslide.open_slide(svs_path)
        print(file_path, len(self.results))
        print('patch_level: ', self.attr['patch_level'], 'patch_size:', self.attr['patch_size'])


    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        coord = self.results[idx]
        patch = self.wsi.read_region((coord[0], coord[1]), self.attr['patch_level'],
                                (self.attr['patch_size'], self.attr['patch_size'])).convert('RGB')
        if transforms:
            patch = self.transform(patch)
        return coord, patch



def extra_feat(file_path, resnet50_path, save_path):
    files = {}

    names = []
    resnet50_extractor = Resnet50Extractor('resnet50', model_path=resnet50_path)

    print(len(os.listdir(file_path)))
    for i in os.listdir(file_path):
        if i[-2:] != 'h5':
            continue
        name = i[:-3]
        names.append(name)
        files[name] = os.path.join(file_path,i)
    print(len(names))
    for name in names:
        tem = {}
        feat_path = os.path.join(save_path, name+'.pt')
        dataset = CustomDataset(files[name])
        data_loader = DataLoader(dataset, batch_size=256, num_workers=32)
        num = 0
        l = len(data_loader)
        torch.save(tem, f'tem.pt')
        with torch.no_grad():
            for coord, img in tqdm(data_loader):
                img = img.to(device)
                features = resnet50_extractor(img).cpu()
                for i in range(len(features)):
                    tem[coord[i]] = features[i]

                # 将更新后的 tem 保存到临时文件
                if len(tem.keys()) // 256 >= len(data_loader) // 10 + 20:
                    existing_tem = torch.load('tem.pt')
                    existing_tem.update(tem)
                    torch.save(existing_tem, 'tem.pt')
                    tem = {}

        # 最后一次更新和保存
        if tem:
            existing_tem = torch.load('tem.pt')
            existing_tem.update(tem)
            torch.save(existing_tem, 'tem.pt')

        final_tem = torch.load('tem.pt')
        torch.save(final_tem, feat_path)






if __name__ == '__main__':
    args = parser.parse_args()
    extra_feat(args.file_path, args.model_path, args.save_path)