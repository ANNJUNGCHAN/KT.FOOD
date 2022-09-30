# Package Loading
### base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### vision
from PIL import Image
import cv2

### path
from glob import glob
import os

# Make Directiory dataframe

def directory_dataframe(data_directory) :
    """
    이미지의 경로를 저장하는 데이터 프레임을 만드는 함수
    ---
    data_directory : class 별로 각각의 폴더에 담긴 이미지가 있는 폴더 경로
                     ('/경로1/경로2/~~~')로 입력해야하며, 마지막에는 / 을 붙이지 않는다.
    ---
    output
    df : 이미지 경로를 저장한 데이터 프레임
    df["dir"] : 이미지 경로
    df["target"] : 이미지의 class
    """
    food_name = []
    for i in glob(data_directory + "/*") :
        food_name.append(i.replace(data_directory + "\\",""))

    dir_list = []
    target = []
    for i in food_name :
        dir_list.append(glob(data_directory + "/" + i + '/*.jpg'))
        for j in range(len(glob(data_directory + "/" + i + '/*.jpg'))) :
            target.append(i)
    dir_list_unit = []
    for j in range(len(dir_list)) :
        for i in dir_list[j] :
            dir_list_unit.append(i)

    df = pd.DataFrame()
    df["dir"] = dir_list_unit
    df["target"] = target

    return df

def RemoveBackground(dir) :
    """
    이미지의 배경을 제거하는 함수
    - 256,256 이미지에 대해서만 적용 가능
    ---
    dir : 이미지 경로
    ---
    output
    image : 배경이 제거된 이미지
    """
    image = plt.imread(dir)

    ### 사각형 좌표 설정
    rectangle = (0,16,256,256)

    ### 초기 마스크 생성
    mask = np.zeros(image.shape[:2],np.uint8)

    ### grabCut에 사용할 임시 배열 설정
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    ### grapCut
    cv2.grabCut(image,
                mask,
                rectangle,
                bgdModel,
                fgdModel,
                10,
                cv2.GC_INIT_WITH_RECT)

    ### 배경인 곳은 0, 그 외에는 1로 설정하는 마스크 생성
    mask_2 = np.where((mask == 2) | (mask==0),0,1).astype('uint8')

    ### 이미지에 새로운 마스크를 곱해 배경을 제거
    image = image * mask_2[:,:,np.newaxis]
    return image