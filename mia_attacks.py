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
from train_mia import load_test_X, test, attack_net

args = parse_args()
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def attack(victim_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1024

    if victim_model is None:
        victim_model = CNN()
        if args.model_fix_flag == 1:
            if args.dataset == 'origin':
                model_dict = torch.load(
                    f'model/{args.dataset}_fix_alpha{args.alpha}_model.ckpt')
            else:
                model_dict = torch.load(
                    f'model/{args.dataset}_fix_alpha{args.alpha}_model_ratio{args.ratio}_seed{args.seed}.ckpt'
                )
        else:
            if args.dataset == 'origin':
                model_dict = torch.load(f'model/origin_model.ckpt')
            else:
                model_dict = torch.load(
                    f'model/{args.dataset}_model_ratio{args.ratio}_seed{args.seed}.ckpt')
        victim_model.load_state_dict(model_dict)

    for name, param in victim_model.named_parameters():
        param.requires_grad = False

    attack_model = torch.load("attack_model.pt")


    with open(f'dataset/unlearn_dataset_{args.ratio}_seed{args.seed}_save.pkl', 'rb') as f:
        unlearn_data = dill.load(f)
    unlearn_loader = data.DataLoader(unlearn_data,
                                        batch_size=len(unlearn_data))
    unlearn_X_shadow = load_test_X(victim_model, unlearn_loader, device)
    unlearn_y_shadow = torch.ones((len(unlearn_data), 1))
    test_tensors = [unlearn_X_shadow, unlearn_y_shadow]
    test_shadow_dataset = TensorDataset(*test_tensors)


    acc = test(attack_model, device,
               DataLoader(test_shadow_dataset, BATCH_SIZE, shuffle=False))
    # print(acc)
    return acc

if __name__=="__main__":
    model = torch.load("ul_model/ul_model.pt")
    atk_acc = attack(model)
    # atk_acc = attack()
    print(f"The attack accuracy is: {atk_acc}")