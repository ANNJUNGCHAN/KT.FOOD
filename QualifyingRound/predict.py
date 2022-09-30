### Base
import numpy as np
import pandas as pd

### directory
from glob import glob

### Keras & Tensorflow
from tensorflow import keras
from keras import Input
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


### Warnings
import warnings
warnings.filterwarnings(action='ignore')

def predict(model,flow_dataset,load_checkpoint_folder) :
    """
    모델 평가를 위한 함수
    - 5KFOLD로 학습한 후, accuracy의 평균을 출력
    ---input---
    model : 불러올 모델의 개형 : countinue.py에서 정의한 create_model()을 불러와 모델의 개형을 정의한 후, 넣어줌
    flow_dataset : 이미지 데이터 경로와 target으로 이루어진 데이터셋(dir,target)
    load_checkpoint_folder : 모델을 불러올 경로(끝에 /를 붙인다)
    ---output---
    print : accuracy
    return : predict한 결과
    """
    # Load images using Keras ImageDataGenerator

    datagen_test = ImageDataGenerator(rescale=1./255)
    test_generator = datagen_test.flow_from_dataframe(
        dataframe=flow_dataset,
        x_col='dir',
        y_col='target',
        batch_size=256,
        seed=7,
        shuffle=False,
        class_mode='categorical',
        target_size=(224,224),
        )
        
    latest = tf.train.latest_checkpoint(load_checkpoint_folder)
    model.load_weights(latest)
        
    results = model.evaluate(test_generator)

    prediction=model.predict(test_generator)

    print(results)

    return prediction
