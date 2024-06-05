import torch
from torch.utils import data
from torch.autograd import Variable
from torchvision.datasets import cifar
import torchvision
from torchvision.datasets import mnist
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utility.parser import parse_args
import dill
import torch.nn.functional as F
from train_model import CNN
import torch.optim as optim
from PIL import Image
from time import *
import numpy as np

args = parse_args()
with open(f'dataset/unlearn_dataset_{args.ratio}_seed{args.seed}_save.pkl',
          'rb') as f:
    unlearn_data = dill.load(f)
with open(f'dataset/remain_dataset_{args.ratio}_seed{args.seed}_save.pkl',
          'rb') as f:
    remain_data = dill.load(f)
data_path = args.data_path

batch_size = args.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
if args.model_fix_flag == 1:
    if args.dataset == 'origin':
        model_dict = torch.load(
            f'model/{args.dataset}_fix_alpha{args.alpha}_model.ckpt')
    else:
        model_dict = torch.load(
            f'model/{args.dataset}_fix_alpha{args.alpha}_model_ratio{args.ratio}.ckpt'
        )
else:
    if args.dataset == 'origin':
        model_dict = torch.load(f'model/origin_model.ckpt')
    else:
        model_dict = torch.load(
            f'model/{args.dataset}_model_ratio{args.ratio}.ckpt')
model.load_state_dict(model_dict)
model.to(device)
unlearn_loader = data.DataLoader(unlearn_data, batch_size=len(unlearn_data))
remain_loader = data.DataLoader(remain_data, batch_size=len(remain_data))
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
begin_time = time()

for idx, (t_data, t_target) in enumerate(remain_loader):
    t_data, t_target = t_data.to(device), t_target.to(device)
    pred = model(t_data)
    loss1 = criterion(pred, t_target)
    loss1.backward()
for idx, (t_data, t_target) in enumerate(unlearn_loader):
    t_data, t_target = t_data.to(device), t_target.to(device)
    pred = model(t_data)  #batch_size*10
    loss2 = -criterion(pred, t_target)
    loss2.backward()
optimizer.step()

end_time = time()
print(end_time - begin_time)

torch.save(model, "ul_model/ul_model.pt")