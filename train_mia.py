import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import dill
from torchvision.datasets import cifar
import torchvision.transforms as transforms
from utility.parser import parse_args
import numpy as np
from train_model import CNN
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, recall_score
import torch.nn.functional as F

args = parse_args()
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class attack_net(nn.Module):

    def __init__(self):
        super(attack_net, self).__init__()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(20, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(32, 1)
        self.fc4 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.drop3(self.relu(self.fc4(x)))
        x = self.sig(self.fc3(x))
        return x


def load_pos_X(model, device, data_size):
    with open(f'dataset/remain_dataset_0.1_save.pkl', 'rb') as f:
        train_data = dill.load(f)
    pos_data, _ = data.random_split(
        train_data, [data_size, len(train_data) - data_size])
    batch_size = 1024

    pos_loader = data.DataLoader(pos_data, batch_size=batch_size)
    pos_shadow_list = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (images, labels) in pos_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)

            onehot_labels = F.one_hot(labels, num_classes=10)
            outputs_shadow = torch.cat((outputs, onehot_labels), dim=1)
            pos_shadow_list.append(outputs_shadow)

    pos_shadow = torch.cat(pos_shadow_list)
    return pos_shadow


def load_neg_X(model, device, data_size):
    with open('dataset/valid_dataset_save.pkl', 'rb') as f:
        test_data = dill.load(f)
    # neg_data = test_data[:data_size]
    neg_data, _ = data.random_split(
        test_data, [data_size, len(test_data) - data_size])
    batch_size = 1024

    neg_loader = data.DataLoader(neg_data, batch_size=batch_size)
    neg_shadow_list = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (images, labels) in neg_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)

            onehot_labels = F.one_hot(labels, num_classes=10)
            outputs_shadow = torch.cat((outputs, onehot_labels), dim=1)
            neg_shadow_list.append(outputs_shadow)

    neg_shadow = torch.cat(neg_shadow_list)
    return neg_shadow


def load_test_X(model, data_loader, device):
    test_shadow_list = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (images, labels) in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)

            onehot_labels = F.one_hot(labels, num_classes=10)
            outputs_shadow = torch.cat(
                (outputs - onehot_labels, onehot_labels), dim=1)
            test_shadow_list.append(outputs_shadow)

    test_shadow = torch.cat(test_shadow_list)
    return test_shadow


def load_shadow_dataloader(model,
                           device,
                           data_size=6000,
                           shuffle=True,
                           batch_size=1024):
    if data_size > 10000:
        print("Data size is larger than test_data in load_neg().")
        return None

    X_pos = load_pos_X(model, device, data_size)
    X_neg = load_neg_X(model, device, data_size)
    data = torch.cat((X_neg, X_pos))
    labels = torch.cat((torch.zeros((data_size, 1)), torch.ones(
        (data_size, 1))))

    tensors = [data, labels]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size, shuffle)


def train(model, device, train_loader, optimizer, criterion, losses):
    model.train()

    for idx, (t_data, t_target) in enumerate(train_loader):
        t_data, t_target = t_data.to(device), t_target.to(device)
        pred = model(t_data)  #batch_size*10
        loss = criterion(pred, t_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            losses.append(loss.item())  


def test(model, device, test_loader):
    model.eval()
    correct = 0  #预测对了几个。
    total = 0
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.data.gt(0.5)
            preds_list.append(predicted)
            labels_list.append(labels)

            total += labels.size(0)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    preds_all = torch.cat(preds_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    preds_all = preds_all.cpu().numpy()
    labels_all = labels_all.cpu().numpy()

    acc = correct / total
    return acc


if __name__ == "__main__":
    BATCH_SIZE = 2048
    LR = 0.01
    DATA_SIZE = 10000
    EPOCHS = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use device {device}")

    victim_model = CNN()
    if args.model_fix_flag == 1:
        model_dict = torch.load(
            f'model/{args.dataset}_fix_alpha{args.alpha}_model_ratio{args.ratio}.ckpt'
        )
    elif args.dataset == 'noised':
        model_dict = torch.load('model/noised_model.ckpt')
    else:
        model_dict = torch.load('model/origin_model.ckpt')

    victim_model.load_state_dict(model_dict)

    for name, param in victim_model.named_parameters():
        param.requires_grad = False

    attack_model = attack_net().to(device)
    optimizer = optim.Adam(attack_model.parameters(), lr=LR)

    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch:1/(epoch+1))

    X_shadow = np.load("shadowX.npy")
    y_shadow = np.load("shadowy.npy")
    shadow_dataloader = DataLoader(
        TensorDataset(*[torch.tensor(X_shadow),
                        torch.tensor(y_shadow)]), 8192,shuffle=True)
    losses = []
    f1es = []
    acces = []
    for epoch in range(EPOCHS):
        train(attack_model, device, shadow_dataloader, optimizer, criterion,
              losses)

    torch.save(attack_model, "attack_model.pt")

    import matplotlib.pyplot as plt
    len_l = len(losses)
    x = [i for i in range(len_l)]
    figure = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x, losses)
    plt.savefig(f"figure/train_mia_loss.png")

    with open(f'dataset/unlearn_dataset_0.1_save.pkl', 'rb') as f:
        unlearn_data = dill.load(f)
    unlearn_loader = data.DataLoader(unlearn_data,
                                     batch_size=len(unlearn_data))
    unlearn_X_shadow = load_test_X(victim_model, unlearn_loader, device)
    unlearn_y_shadow = torch.ones((len(unlearn_data), 1))
    data_path = args.data_path
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    unuse_data = cifar.CIFAR10(data_path,
                               train=False,
                               transform=transform,
                               download=True)
    unuse_data, _ = data.random_split(
        unuse_data, [len(unlearn_data),
                     len(unuse_data) - len(unlearn_data)])
    unuse_loader = data.DataLoader(unuse_data, batch_size=len(unuse_data))
    unuse_X_shadow = load_test_X(victim_model, unuse_loader, device)
    unuse_y_shadow = torch.zeros((len(unuse_data), 1))
    data_test = torch.cat((unlearn_X_shadow, unuse_X_shadow))
    label_test = torch.cat((unlearn_y_shadow, unuse_y_shadow))
    test_tensors = [data_test, label_test]
    test_shadow_dataset = TensorDataset(*test_tensors)
    acc_all = test(attack_model, device,
                   DataLoader(test_shadow_dataset, BATCH_SIZE, shuffle=False))
    unlearn_tensors = [unlearn_X_shadow, unlearn_y_shadow]
    unlearn_shadow_dataset = TensorDataset(*unlearn_tensors)
    acc_use = test(
        attack_model, device,
        DataLoader(unlearn_shadow_dataset, BATCH_SIZE, shuffle=False))
    unuse_tensors = [unuse_X_shadow, unuse_y_shadow]
    unuse_shadow_dataset = TensorDataset(*unuse_tensors)
    acc_unuse = test(
        attack_model, device,
        DataLoader(unuse_shadow_dataset, BATCH_SIZE, shuffle=False))

    print(f"f1_all is {acc_all}, f1_use is {acc_use}, f1_unuse is {acc_unuse}")
