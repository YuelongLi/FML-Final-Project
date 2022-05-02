from autoattack import AutoAttack
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from models.wideresnet import WideResNet
import os

# settings
device = torch.device("cpu")
kwargs = {}


# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, **kwargs)

model = WideResNet()
model.load_state_dict(torch.load('/Users/olgavrou/Documents/NYU/Project/auto-attack/model_cifar_wrn.pt', map_location=torch.device('cpu')))
model.eval()


ind = 0

adversary = AutoAttack(model, norm='Linf', eps=float(8/255), version='standard')

data_loaders = [train_loader, test_loader]
data_points = 1
out_dir = ['adv_train_dat', 'adv_dat']

for i, loader in enumerate(data_loaders):
    if not os.path.exists(out_dir[i]):
        os.makedirs(out_dir[i])
    quota_reached = False
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        print(f"bi: {batch_idx}")

        if quota_reached:
            break
        for x_adv in adversary.generate_adversarial_data(data, target, bs=1):
            data = x_adv
        
            torch.save(data, f'{out_dir[i]}/adversarial-data_{batch_idx}.pt')
            ind = ind + 1
            if ind > data_points:
                quota_reached = True
