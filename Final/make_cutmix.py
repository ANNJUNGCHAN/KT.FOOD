# Load Package

### Directory
import os
from glob import glob

### BASE
import pandas as pd
import numpy as np

### VISUALIZATION
import cv2 as cv
import matplotlib.pyplot as plt

### TQDM
from tqdm.auto import tqdm

### WRANING
import warnings
warnings.filterwarnings(action='ignore')

### RANDOM
import random

# make directory and target DataFrame
folder_name = os.listdir("/home/work/team06/glico-learning-small-sample/data/Food_dataset/train/")
train_dir = "/home/work/team06/glico-learning-small-sample/data/Food_dataset/train/"

train = pd.DataFrame()
train["dir"] = 0
for i in folder_name :
    temp = pd.DataFrame()
    temp["dir"] = glob(train_dir + i +"/" + "*")
    train = pd.concat([train,temp],axis = 0)

folder_name = os.listdir("/home/work/team06/glico-learning-small-sample/data/Food_dataset/val/")
valid_dir = "/home/work/team06/glico-learning-small-sample/data/Food_dataset/val/"

valid = pd.DataFrame()
valid["dir"] = 0
for i in folder_name :
    temp = pd.DataFrame()
    temp["dir"] = glob(valid_dir + i +"/" + "*")
    valid = pd.concat([valid,temp],axis = 0)

train["target"] = train["dir"].str.split("/").apply(lambda x : x[-2])
valid["target"] = valid["dir"].str.split("/").apply(lambda x : x[-2])

train.reset_index(drop = True, inplace = True)
valid.reset_index(drop = True, inplace = True)

for i in train["target"].unique() :
    os.makedirs("/home/work/team06/CutMix/train/" + i)

target_list = train["target"].unique()



# CUTMIX

### Define Bounding Box
def rand_bbox(size, lam):
    
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    ratio = abs(bbx1-bbx2)*abs(bby1-bby2) / (128*128)

    return bbx1, bby1, bbx2, bby2, ratio

### Define CUTMIX Function
def cut_generator(train_img,save_dir):

    rand_ind = int(random.uniform(0,len(train_img)))
    rand_ind2 = int(random.uniform(0,len(train_img)))

    img1 = cv.imread(train_img["dir"].iloc[rand_ind]).copy()
    img2 = cv.imread(train_img["dir"].iloc[rand_ind2])
    
    bbx1,bby1,bbx2,bby2,ratio = rand_bbox([128,128],np.random.beta(1,1)) # beta(alpha,alpha) alpha = 1

    bbx1,bbx2,bby1,bby2 = min(bbx1,bbx2),max(bbx1,bbx2),min(bby1,bby2),max(bby1,bby2)

    img1_o = img1.copy() # 원본
    img1[bbx1:bbx2,bby1:bby2,:] = img2[bbx1:bbx2,bby1:bby2,:]
    img1_c = img1.copy()

    img1 = img1_o.copy()
    image = img1_c
    cv.imwrite(save_dir,image)

### Run CUTMIX
for i in tqdm(target_list) : 
    temp = train.loc[train["target"] == i]
    print(i + " : 시작")
    for j in tqdm(range(500)) :
        cut_generator(temp,"/home/work/team06/CutMix/train/" + i + "/" + i + "_" + str(j) + ".jpg")