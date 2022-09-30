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
    모델을 이용하여 예측하기 위한 함수
    ---input---
    model : 불러올 모델의 개형 : countinue.py에서 정의한 create_model()을 불러와 모델의 개형을 정의한 후, 넣어줌
    flow_dataset : 이미지 데이터 경로와 target으로 이루어진 데이터셋(dir,target)
    load_checkpoint_folder : 모델을 불러올 경로(끝에 /를 붙인다)
    ---output---
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

def result_predict(prediction_result) :
    """
    predict한 결과를 원래의 target으로 변환
    ---input---
    prediction_result : predict한 결과(1차원 벡터)
    ---output---
    predict_list : 원래의 target으로 변환한 결과
    """
    argmax_pred = np.argmax(prediction_result,axis = 1)
    
    config = {
    '가자미전': 0,
    '간장게장': 1,
    '감자탕': 2,
    '거봉포도': 3,
    '고구마': 4,
    '고구마맛탕': 5,
    '고등어찌개': 6,
    '곱창구이': 7,
    '군만두': 8,
    '굴전': 9,
    '김치찌개': 10,
    '깻잎나물볶음': 11,
    '꼬리곰탕': 12,
    '꽈리고추무침': 13,
    '나시고랭': 14,
    '누룽지': 15,
    '단무지': 16,
    '달걀말이': 17,
    '달걀볶음밥': 18,
    '달걀비빔밥': 19,
    '닭가슴살': 20,
    '닭개장': 21,
    '닭살채소볶음': 22,
    '닭칼국수': 23,
    '도가니탕': 24,
    '도토리묵': 25,
    '돼지감자': 26,
    '돼지고기구이': 27,
    '두부': 28,
    '두부고추장조림': 29,
    '딸기': 30,
    '떡갈비': 31,
    '떡국': 32,
    '레드와인': 33,
    '마늘쫑무침': 34,
    '마카롱': 35,
    '매운탕': 36,
    '미소된장국': 37,
    '미소장국': 38,
    '미역초무침': 39,
    '바나나우유': 40,
    '바지락조개국': 41,
    '보리밥': 42,
    '불고기': 43,
    '비빔밥': 44,
    '뼈해장국': 45,
    '삼선자장면': 46,
    '새우매운탕': 47,
    '새우볶음밥': 48,
    '생연어': 49
    }

    predict_list = []
    for i in argmax_pred :
        predict_list.append(list(config.keys())[i])
    return predict_list
