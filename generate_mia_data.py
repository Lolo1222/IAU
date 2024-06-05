import numpy as np

from absl import flags
from absl import app

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.datasets import cifar  # 获取数据集和数据预处理
import torch.nn.functional as F

# from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from utility.parser import parse_args
from skorch import NeuralNetClassifier
from mia.wrappers import TorchWrapper
from resnet import ResNet18
import dill
import torch.optim as optim  #优化器
from PIL import Image
import copy

class attack_CNN(nn.Module):

    def __init__(self):
        super(attack_CNN, self).__init__()
        self.fc1 = nn.Linear(20, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        return torch.sigmoid(x)


def prepare_data(model, data_in, data_out):
    """
    Prepare the data in the attack model format.

    :param model: Classifier
    :param (X, y) data_in: Data used for training
    :param (X, y) data_out: Data not used for training

    :returns: (X, y) for the attack classifier
    """
    NUM_CLASSES = 10
    X_in, y_in = data_in
    X_out, y_out = data_out
    device = 'cuda'
    model.to(device)
    import torch.nn.functional as F
    y_in_one_hots = y_in
    y_out_one_hots = y_out
    y_hat_in = F.softmax(model(X_in.to(device)), dim=1)
    y_hat_out = F.softmax(model(X_out.to(device)), dim=1)
    y_hat_in = y_hat_in.detach().cpu() - y_in_one_hots
    y_hat_out = y_hat_out.detach().cpu() - y_out_one_hots

    labels = np.ones(y_in.shape[0])
    labels = np.hstack([labels, np.zeros(y_out.shape[0])])

    data = np.c_[y_hat_in, y_in_one_hots]
    data = np.vstack([data, np.c_[y_hat_out, y_out_one_hots]])
    return data, labels


def train(model,
          device,
          train_loader,
          optimizer,
          criterion=nn.BCELoss(),
          losses=None):
    model.train()

    for idx, (t_data, t_target) in enumerate(train_loader):
        t_data, t_target = t_data.to(device), t_target.to(device)
        pred = model(t_data)  #batch_size*10
        loss = criterion(pred, t_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if losses is not None and idx % 5 == 0:
            losses.append(loss.item())


def test(model, device, test_loader):
    model.eval()
    correct = 0
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


def validate(model, device, loader):
    total = 0
    correct = 0
    for imgs, labels in loader:
        batch_size = len(imgs)
        total += batch_size
        imgs, labels = imgs.to(device), labels.to(device)
        out_probs = model(imgs)
        out = torch.argmax(out_probs, dim=1)
        labels = labels.view(out.shape)
        correct += torch.sum(out == labels)
    return correct / total


class SMB():

    def __init__(self,
                 target_epochs,
                 criterion,
                 optimizer,
                 lr,
                 shadow_dataset_size,
                 num_models,
                 seed=42):
        self.target_epochs = target_epochs
        self.criterion = criterion
        self.opt = optimizer
        self.lr = lr
        self.sds = shadow_dataset_size
        self.num_models = num_models
        self.seed = seed

    def prepare_data(self, model, data_in, data_out):
        """
        Prepare the data in the attack model format.

        :param model: Classifier
        :param (X, y) data_in: Data used for training
        :param (X, y) data_out: Data not used for training

        :returns: (X, y) for the attack classifier
        """
        NUM_CLASSES = 10
        X_in, y_in = data_in
        X_out, y_out = data_out
        device = 'cuda'
        import torch.nn.functional as F

        y_in_one_hots = F.one_hot(y_in, num_classes=NUM_CLASSES)
        y_out_one_hots = F.one_hot(y_out, num_classes=NUM_CLASSES)

        y_hat_in = F.softmax(model(X_in.to(device)), dim=1)
        y_hat_out = F.softmax(model(X_out.to(device)), dim=1)
        y_hat_in = y_hat_in.detach().cpu() - y_in_one_hots
        y_hat_out = y_hat_out.detach().cpu() - y_out_one_hots

        labels = np.ones(y_in.shape[0])
        labels = np.hstack([labels, np.zeros(y_out.shape[0])])

        data = np.c_[y_hat_in, y_in_one_hots]
        data = np.vstack([data, np.c_[y_hat_out, y_out_one_hots]])
        return data, labels

    def fit_transform(self, train_data):
        self._fit(train_data)
        return self._transform()

    def _fit(self, train_data):
        self.data = train_data
        self.shadow_models_ = []
        device = 'cuda'
        self.shadow_train_indices_ = []
        self.shadow_test_indices_ = []

        indices = np.arange(len(self.data))

        for num in range(self.num_models):
            print("model num:", num)
            # Pick indices for this shadow model.
            shadow_indices = np.random.choice(indices,
                                              2 * self.sds,
                                              replace=False)
            train_indices = shadow_indices[:self.sds]
            test_indices = shadow_indices[self.sds:]
            self.shadow_train_indices_.append(train_indices)
            self.shadow_test_indices_.append(test_indices)

            # Train the shadow model.
            shadow_model = ResNet18().to(device)
            opt = self.opt(shadow_model.parameters(), self.lr)
            sampler = SubsetRandomSampler(train_indices)
            shadow_loader = DataLoader(self.data,
                                       batch_size=1024,
                                       sampler=sampler)
            for ep in range(self.target_epochs):
                train(shadow_model, device, shadow_loader, opt, self.criterion)

            self.shadow_models_.append(shadow_model)

    def _transform(self):
        '''Produce in/out data for training the attack model.'''
        shadow_data_array = []
        shadow_label_array = []

        for i in range(self.num_models):
            shadow_model = self.shadow_models_[i]
            train_indices = self.shadow_train_indices_[i]
            test_indices = self.shadow_test_indices_[i]
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            train_loader = DataLoader(self.data,
                                      batch_size=self.sds,
                                      sampler=train_sampler)
            test_loader = DataLoader(self.data,
                                     batch_size=self.sds,
                                     sampler=test_sampler)
            for (ima_train, label_train) in train_loader:
                train_data = ima_train, label_train
            for (ima_test, label_test) in test_loader:
                test_data = ima_test, label_test
            shadow_data, shadow_labels = self.prepare_data(
                shadow_model, train_data, test_data)

            shadow_data_array.append(shadow_data)
            shadow_label_array.append(shadow_labels)

        X_transformed = np.vstack(shadow_data_array).astype("float32")
        y_transformed = np.hstack(shadow_label_array).astype("float32")
        return X_transformed, y_transformed


args = parse_args()
data_path = args.data_path
transform = transforms.Compose([  #定义transform
    transforms.ToTensor(),  #将图片由numpy类型转换为tensor类型，torch中数据类型为张量，tensor类型
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))  #归一化，0.5可自定义，范围在【-1，1】、
])
test_data = cifar.CIFAR10(data_path,
                          train=True,
                          transform=transform,
                          download=True)

model = ResNet18()
if args.model_fix_flag == 1:
    model_dict = torch.load(
        f'model/{args.dataset}_fix_alpha{args.alpha}_model.ckpt')
elif args.dataset == 'noised':
    model_dict = torch.load('model/noised_model.ckpt')
else:
    model_dict = torch.load('model/origin_model.ckpt')
model.load_state_dict(model_dict)
test_model = TorchWrapper(module=model,
                          criterion=nn.CrossEntropyLoss(),
                          optimizer=optim.Adam,
                          enable_cuda=False,
                          optimizer_params={'lr': 0.001})

def demo(argv):
    del argv  # Unused.
    SHADOW_DATASET_SIZE = int(len(test_data) * 0.45)
    NUM_CLASSES = 10

    # Train the shadow models.
    FLAGS = flags.FLAGS
    flags.DEFINE_integer(
        "target_epochs", 25,
        "Number of epochs to train target and shadow models.")
    flags.DEFINE_integer("attack_epochs", 20,
                         "Number of epochs to train attack models.")
    flags.DEFINE_integer("num_shadows", 3,
                         "Number of epochs to train attack models.")

    smb = SMB(FLAGS.target_epochs, nn.CrossEntropyLoss(), optim.Adam, 0.001,
              SHADOW_DATASET_SIZE, FLAGS.num_shadows)

    attacker_train, attacker_test = train_test_split(test_data, test_size=0.1)

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(attacker_train)
    testset = cifar.CIFAR10(data_path,
                            train=False,
                            transform=transform,
                            download=True)
    test_loader = DataLoader(testset, 1024)
    for num in range(smb.num_models):
        acc = validate(smb.shadow_models_[num], 'cuda', test_loader)
        print("acc is ", acc)

    y_shadow = np.expand_dims(y_shadow, 1)
    np.random.seed(2)
    indices = np.arange(len(y_shadow))
    np.random.shuffle(indices)
    X_shadow = X_shadow[indices]
    y_shadow = y_shadow[indices]
    np.save("shadowX.npy", X_shadow)
    np.save("shadowy.npy", y_shadow)



if __name__ == "__main__":
    app.run(demo)