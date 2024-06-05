import torch
from torch.utils import data  # 获取迭代数据
from torch.autograd import Variable  # 获取变量
import torchvision
from torchvision.datasets import cifar  # 获取数据集和数据预处理
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utility.parser import parse_args
import dill
from PIL import Image
import copy
import torch.nn as nn  #网络结构
import torch.nn.functional as F
import torch.optim as optim  #优化器
from time import *
# from torch.optim.lr_scheduler import StepLR

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu((self.conv2(x))))
        x = x.view(-1, 16*5*5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train(model, device, train_loader, optimizer, alpha, losses, model_fix_flag=0):
    model.train()

    criterion = nn.CrossEntropyLoss()

    for idx, (t_data, t_target) in enumerate(train_loader):
        t_data, t_target = t_data.to(device), t_target.to(device)
        pred = model(t_data)  #batch_size*10
        temp_loss = criterion(pred, t_target)
        if model_fix_flag == 0:
            loss = temp_loss
        else:
            grad_tuple = torch.autograd.grad(temp_loss,
                                             model.parameters(),
                                             create_graph=True)
            loss = temp_loss + alpha * sum(
                [torch.norm(para_grad) for para_grad in grad_tuple])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            losses.append(loss.item()) 


def test(model, device, test_loader):
    model.eval()
    correct = 0 
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            numbers, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    batch_size = args.batch_size
    lr = args.lr 
    alpha = args.alpha
    model_fix_flag = args.model_fix_flag
    data_path = args.data_path
    num_epochs = args.epoch
    typemodel = "origin" if model_fix_flag==0 else "fix"
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    if args.dataset == 'origin':
        with open('dataset/train_dataset_save.pkl', 'rb') as f:
            train_data = dill.load(f)
    elif args.dataset == 'remain':
        with open(f'dataset/remain_dataset_{args.ratio}_seed{args.seed}_save.pkl', 'rb') as f:
            train_data = dill.load(f)
    elif args.dataset == 'noised':
        with open(f'dataset/noised_full_dataset_{args.ratio}_save.pkl', 'rb') as f:
            train_data = dill.load(f)
    else:
        print("Unknown dataset type!")

    test_data = cifar.CIFAR10(data_path,
                              train=False,
                              transform=transform,
                              download=True)
    with open('dataset/valid_dataset_save.pkl', 'rb') as f:
        valid_data = dill.load(f)

    train_loader = data.DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=False)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = [] 
    accuracys = []
    t_accs = []
    count = 0

    begin_time = time()  
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, alpha, losses,
              model_fix_flag)
        acc = test(model, device, valid_loader)
        t_acc = test(model, device, test_loader)
        if epoch == 0:
            best_acc = acc
        else:
            if acc < best_acc:
                count += 1
            else:
                best_acc = acc
                count = 0
                model_copy = copy.deepcopy(model)

        accuracys.append(acc)
        t_accs.append(t_acc)
        if count > 9:
            print(f"Early stop at epoch {epoch},")
            break
    end_time = time()

    print(f"{end_time-begin_time}")
    print(f"Valid accuracy is {best_acc},")

    if model_fix_flag == 0:
        if args.dataset=='origin':
            torch.save(model_copy.state_dict(), f'model/origin_model.ckpt')

        else:
            torch.save(model_copy.state_dict(), f'model/{args.dataset}_model_ratio{args.ratio}_seed{args.seed}.ckpt')
    else:
        if args.dataset == 'origin':
            torch.save(
                model_copy.state_dict(),
                f'model/{args.dataset}_fix_alpha{alpha}_model.ckpt'
        )
        else:
            torch.save(
                model_copy.state_dict(),
                f'model/{args.dataset}_fix_alpha{alpha}_model_ratio{args.ratio}_seed{args.seed}.ckpt'
        )