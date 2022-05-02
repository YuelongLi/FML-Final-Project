from xmlrpc.client import FastUnmarshaller
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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=FastUnmarshaller, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, **kwargs)

model = WideResNet()
model.load_state_dict(torch.load('/Users/olgavrou/Documents/NYU/Project/auto-attack/model_cifar_wrn.pt', map_location=torch.device('cpu')))

model.eval()
import math
import numpy as np

outputs = ['train.dat', 'test.dat']
input_dirs = ['adv_train_dat', 'adv_dat']

named_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

for active_class in range(0,10):
    # going to pick that active_class is +1 and anything else is -1

    for i, outfile in enumerate(outputs):
        total = 0
        acc = 0
        datapoints = len(os.listdir(input_dirs[i]))
        with open("libsvm_" + named_classes[active_class] + "_" + outfile, 'w') as libsvm_data:
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                # print(f"bi: {batch_idx}")
                if batch_idx >= datapoints:
                    break
                x_adv = torch.tensor(torch.load(f'{input_dirs[i]}/adversarial-data_{batch_idx}.pt'))
                data = x_adv

                pred_probab = model(data)
                y_pred1 = pred_probab.argmax(1)
                label = target.tolist()[0]

                # get the accuracy of the model on the cat label (as if we are doing binary classification)
                total = total + 1
                if (label == active_class and label == y_pred1.tolist()[0]) or (label != active_class and y_pred1.tolist()[0] != active_class):
                    acc = acc + 1

                pp = pred_probab.tolist()[0]
                libsvm_label = None
                if label == active_class:
                    libsvm_label = "+1"
                else:
                    libsvm_label = "-1"

                label_features = f"{libsvm_label} 1:{pp[0]} 2:{pp[1]} 3:{pp[2]} 4:{pp[3]} 5:{pp[4]} 6:{pp[5]} 7:{pp[6]} 8:{pp[7]} 9:{pp[8]} 10:{pp[9]}"

                libsvm_data.write(label_features)
                libsvm_data.write("\n")
        if 'train' in input_dirs[i]:
            print(f"TRADES nn model accuracy for {named_classes[active_class]} class: {(float(acc)/float(total)) * 100}%")
