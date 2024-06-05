from torchvision.datasets import cifar  # 获取数据集和数据预处理
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dill
from utility.parser import parse_args
from PIL import Image
import torch


args = parse_args()
np.random.seed(args.seed)


class MyDataSet(Dataset):

    def __init__(self, data, targets, transform=transforms.ToTensor()):
        self.data = data

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


data_path = args.data_path
valid_size = 0.2

transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)) 
])

with open('dataset/train_dataset_save.pkl', 'rb') as f:
    whole_data = dill.load(f)
ratio = args.ratio  # Unlearn ratio
n = len(whole_data.targets)
indices = np.arange(n)
np.random.shuffle(indices)
targets_array = np.array(whole_data.targets)

unlearn_idx = indices[:int(n * ratio)]
remain_idx = indices[int(n * ratio):]

# Unlearn part
unlearn_targets = list(targets_array[unlearn_idx])
unlearn_data = whole_data.data[unlearn_idx]
unlearn_dataset = MyDataSet(unlearn_data, unlearn_targets,transform=transform)
with open(f'dataset/unlearn_dataset_{ratio}_seed{args.seed}_save.pkl', 'wb') as f:
    dill.dump(unlearn_dataset, f)

# Remain part
remain_targets = list(targets_array[remain_idx])
remain_data = whole_data.data[remain_idx]
remain_dataset = MyDataSet(remain_data, remain_targets, transform=transform)
with open(f'dataset/remain_dataset_{ratio}_seed{args.seed}_save.pkl', 'wb') as f:
    dill.dump(remain_dataset, f)

