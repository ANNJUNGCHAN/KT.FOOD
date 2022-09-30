### Load Package

# data handling
import copy
import numpy as np
import pandas as pd

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as Datasets

# tqdm for bar
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# base packages
import time
from glob import glob

# fix seed
import random
import os
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(41) 

# filter warnings
import warnings
warnings.filterwarnings(action='ignore')

### MODELING

# CONFIG
CFG = {
    'IMG_SIZE':256,
    'EPOCHS':1000,
    'LEARNING_RATE':1e-7,
    'BATCH_SIZE':32,
    'SEED':41
}

# INPUT TEST FOLDER
test_path = "/home/work/team06/TestFolder/data/val"

# MODEL Define

class VGG(nn.Module) :
    def __init__(self, features, Output_dim) :
        super().__init__()
        self.features = features
        self.avgpool= nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096,Output_dim)
        )
    def forward(self, x) :
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0],-1)
        x = self.classifier(h)
        return x, h

vgg11_config = [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']

def get_vgg_layers(config, batch_norm) :
    
    layers = []
    in_channels = 3

    for c in config :
        assert c == 'M' or isinstance(c, int)
        if c == 'M' :
            layers += [nn.MaxPool2d(kernel_size=2)]
        else :
            conv2d = nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1)
            if batch_norm :
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            else :
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = c

    return nn.Sequential(*layers)

vgg11_layers = get_vgg_layers(vgg11_config, batch_norm=True)

OUTPUT_DIM = 100
model = VGG(vgg11_layers, OUTPUT_DIM)

# define tranform
test_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
])


# define DataLoader
test_dataset = torchvision.datasets.ImageFolder(
    test_path,
    transform = test_transforms
)
BATCH_SIZE = CFG['BATCH_SIZE']
test_iterator = data.DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE)

### Output
model = torch.load("/home/work/team06/TestFolder/VGG/checkpoint_1e-7_ep20/VGG-model_12):vl:0.656:tl:0.079.pt")

# Top1 & TOP5 ACCURACY

def accuracy(model, test_data, batch_size=32, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    model.eval()
    top1, top5 = [], []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_data)):
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred, _ = model(x)

            maxk = max(topk)
            batch_size = y.size(0)
            # target = validate_loader_consistency(batch_size, idx, target, testdata)
            _, pred = y_pred.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(y.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul(100.0 / batch_size))
            top1.append(res[0].item())
            top5.append(res[1].item())
    top1_avg = sum(top1) / len(top1)
    top5_avg = sum(top5) / len(top5)
    return top1_avg, top5_avg

# print
top1, top5 =  accuracy(model,test_iterator)
print(f'top1 : {top1} , top5 : {top5}')