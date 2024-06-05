from torchvision.datasets import cifar
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
whole_data = cifar.CIFAR10(data_path,
                           train=True,
                           transform=transform,
                           download=True)
test_data = cifar.CIFAR10(data_path,
                           train=False,
                           transform=transform,
                           download=True)

n = len(whole_data.targets)
indices = np.arange(n)
np.random.shuffle(indices)
targets_array = np.array(whole_data.targets)

train_idx = indices[:n - int(n * valid_size)]
valid_idx = indices[n - int(n * valid_size):]

# Valid part
valid_targets = list(targets_array[valid_idx])
valid_data = whole_data.data[valid_idx]
valid_dataset = MyDataSet(valid_data, valid_targets,transform=transform)
with open('dataset/valid_dataset_save.pkl', 'wb') as f:
    dill.dump(valid_dataset, f)


# Train part
train_targets = list(targets_array[train_idx])
train_data = whole_data.data[train_idx]
train_dataset = MyDataSet(train_data, train_targets, transform=transform)
with open('dataset/train_dataset_save.pkl', 'wb') as f:
    dill.dump(train_dataset, f)
