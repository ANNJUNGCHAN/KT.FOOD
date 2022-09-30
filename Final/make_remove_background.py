### LOAD PACKAGE

# BASE
import numpy as np
import pandas as pd

# VISUALIZATION
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# PATH
from glob import glob
import os

### make directory and target DataFrame
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

### Remove Background Function
def remove_bg(dir,save_dir) :
    image = cv2.imread(dir)

    # 사각형 좌표 설정
    rectangle = (0,16,256,256)

    # 초기 마스크 생성
    mask = np.zeros(image.shape[:2],np.uint8)

    # grabCut에 사용할 임시 배열 설정
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    # grapCut
    cv2.grabCut(image,
                mask,
                rectangle,
                bgdModel,
                fgdModel,
                10,
                cv2.GC_INIT_WITH_RECT)

    # 배경인 곳은 0, 그 외에는 1로 설정하는 마스크 생성
    mask_2 = np.where((mask == 2) | (mask==0),0,1).astype('uint8')

    # 이미지에 새로운 마스크를 곱해 배경을 제거
    image_rgb_nobg = image * mask_2[:,:,np.newaxis]
    image = cv2.cvtColor(image_rgb_nobg, cv2.COLOR_BGR2RGB)
    # 이미지 저장
    cv2.imwrite(save_dir,image_rgb_nobg)

### RUN
for i in train["target"].unique() :
    os.makedirs("/home/work/team06/RAG-VGGs/nobackground/train/" + i)

for i in range(len(train)) :
    dir_make = "/home/work/team06/RAG-VGGs/nobackground/train/" + train["target"].iloc[i] + "/" + train["dir"].iloc[i].split("/")[-1]
    remove_bg(train["dir"][i],dir_make)

train = pd.read_csv("/home/work/team06/VGG_Sample/train_removed_list.csv").drop(columns = "Unnamed: 0")

for i in range(len(train)) :
    dir_make = "/home/work/team06/RAG-VGGs/nobackground/train/" + train["target"].iloc[i] + "/" + train["dir"].iloc[i].split("/")[-1]
    remove_bg(train["dir"][i],dir_make)