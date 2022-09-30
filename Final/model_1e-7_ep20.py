##### Continue Training

### Package Loading

# basic packages
import copy
import numpy as np
import pandas as pd
import time
from tqdm.auto import tqdm
from glob import glob

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as Datasets

# gpu setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix seed
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

### Modeling

# CONFIG
CFG = {
    'IMG_SIZE':256,
    'EPOCHS':1000,
    'LEARNING_RATE':1e-7,
    'BATCH_SIZE':32,
    'SEED':41
}

# Define VGG
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

# vgg13 config
vgg13_config = [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M']

# Define vgg layers
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

vgg13_layers = get_vgg_layers(vgg13_config, batch_norm=True)

# Define Model
OUTPUT_DIM = 100
model = torch.load("/home/work/team06/TestFolder/VGG/checkpoint_1e-4/VGG-model_20):vl:0.712:tl:0.158.pt")
model.eval()

### Data

# Define Data Transform
train_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
])

# Define DataLoad
train_path = "/home/work/team06/TestFolder/data/train" # we use all dataset(Origin + CutMix + BackgroudRemove)
test_path = "/home/work/team06/TestFolder/data/val" # validation not to be changing

train_dataset = torchvision.datasets.ImageFolder(
    train_path,
    transform = train_transforms
)

test_dataset = torchvision.datasets.ImageFolder(
    test_path,
    transform = test_transforms
)

### Training

# Define iteration
BATCH_SIZE = CFG['BATCH_SIZE']
train_iterator = data.DataLoader(
    train_dataset,
    shuffle = True,
    batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE)

# Define Optimizing
optimizer = optim.Adam(model.parameters(), lr = CFG['LEARNING_RATE'])
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

# Define Score
def caculate_accuracy(y_pred, y) :
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum() # 예측이 정답과 일치하는 경우, 그 개수의 합을 correct 변수에 저장
    acc = correct.float() / y.shape[0]
    return acc

# Define Training
def train(model, iterator, optimizer, criterion, device) :
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x,y) in tqdm(iterator) :
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)
        loss = criterion(y_pred, y)
        acc = caculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)

# Define Evaluate
def evalutate(model, iterator, criterion, device) :
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad() :
        for (x,y) in tqdm(iterator) :
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = criterion(y_pred,y)
            acc = caculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Define training time for epoch
def epoch_time(start_time, end_time) :
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Run
EPOCHS = CFG['EPOCHS']
best_valid_loss = float('inf')
for epoch in range(EPOCHS) :
    start_time = time.monotonic()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evalutate(model, test_iterator, criterion, device)

    if valid_loss < best_valid_loss :
        best_valid_loss = valid_loss
        torch.save(model, f"/home/work/team06/TestFolder/VGG/checkpoint_1e-7_ep20/VGG-model_{epoch+1}):vl:{valid_loss:.3f}:tl:{train_loss:.3f}.pt")
    
    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch:{epoch + 1:02} | Epoch Time : {epoch_mins}m {epoch_secs}s')
    print(f'\t Train Loss : {train_loss:.3f} | Train Acc : {train_acc*100:.2f}%')
    print(f'\t Valid Loss : {valid_loss:.3f} | Valid Acc : {valid_acc*100:.2f}%')