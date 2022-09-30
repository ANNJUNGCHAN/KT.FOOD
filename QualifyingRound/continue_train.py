### Base
import numpy as np
import pandas as pd

### directory
from glob import glob

### Keras & Tensorflow
from tensorflow import keras
from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
import tensorflow as tf

### KFOLD
from sklearn.model_selection import StratifiedKFold

### Warnings
import warnings
warnings.filterwarnings(action='ignore')

def create_model() :
    """
    모델을 정의하는 함수
    - VGGs 모델의 개형만을 로딩한다
    """
    input_train = keras.layers.Input(shape = (224,224,3), dtype = 'float32', name = 'input_train')

    conv32_1 = Conv2D(32,4,padding = 'same',activation = 'swish')(input_train)
    conv32_2 = Conv2D(32,3,padding = 'same',activation = 'swish')(conv32_1)
    batch1 = keras.layers.BatchNormalization()(conv32_2)
    maxpool1 = MaxPooling2D(pool_size = (4,4))(batch1)


    conv64_1 = Conv2D(64,3,padding = 'same',activation = 'swish')(maxpool1)
    conv64_2 = Conv2D(64,3,padding = 'same',activation = 'swish')(conv64_1)
    batch2 = keras.layers.BatchNormalization()(conv64_2)
    maxpool2 = MaxPooling2D(pool_size = (4,4))(batch2)

    flat = keras.layers.Flatten()(maxpool2)

    dense1 = keras.layers.Dense(512, activation = 'swish')(flat)
    dropout1 = keras.layers.Dropout(0.3)(dense1)

    dense2 = keras.layers.Dense(256,activation = 'swish')(dropout1)
    dropout2 = keras.layers.Dropout(0.3)(dense2)
    dense4 = keras.layers.Dense(50, activation = "softmax")(dropout2)

    model = Model(input_train, dense4)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=["acc"])

    return model


def countinue_train(model,flow_dataset,load_checkpoint_folder,save_checkpoint_folder,epochs) :
    """
    모델 평가를 위한 함수
    - 5KFOLD로 학습한 후, accuracy의 평균을 출력
    ---input---
    model : 불러올 모델의 개형 : countinue.py에서 정의한 create_model()을 불러와 모델의 개형을 정의한 후, 넣어줌
    flow_dataset : 이미지 데이터 경로와 target으로 이루어진 데이터셋(dir,target)
    load_checkpoint_folder : 모델을 불러올 경로(끝에 /를 붙인다)
    save_checkpoint_folder : 모델을 저장할 경로(끝에 /를 붙인다)
    epochs : 학습할 epoch 수
    ---output---
    5KFOLD의 accuracy 평균
    """
    
    y = flow_dataset["target"]

    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    validation_accuracy = []
    validation_loss = []
    val_acc = 0

    fold_var = 1 # 모델 저장용

    for train_index, val_index in skf.split(flow_dataset, flow_dataset["target"]) :

        train_data = flow_dataset.iloc[train_index]
        validation_data = flow_dataset.iloc[val_index]
        train_data.reset_index(drop = True, inplace = True)

        # Load images using Keras ImageDataGenerator

        datagen_train = ImageDataGenerator(rescale=1./255,
                                        rotation_range=30,
                                            shear_range=0.2,
                                            zoom_range=0.4,
                                            horizontal_flip=True)
        train_generator = datagen_train.flow_from_dataframe(
            dataframe=train_data,
            x_col='dir',
            y_col='target',
            batch_size=256,
            seed=7,
            shuffle=True,
            class_mode='categorical',
            target_size=(224,224),
        )
            

        datagen_test = ImageDataGenerator(rescale=1./255)
        test_generator = datagen_test.flow_from_dataframe(
            dataframe=validation_data,
            x_col='dir',
            y_col='target',
            batch_size=256,
            seed=7,
            shuffle=True,
            class_mode='categorical',
            target_size=(224,224),
        )
        
        model = create_model()
        latest = tf.train.latest_checkpoint(load_checkpoint_folder)
        model.load_weights(latest)
        
            
        checkpoint_path = save_checkpoint_folder + str(fold_var) + ".ckpt"
        modelcheckpoint = ModelCheckpoint(checkpoint_path, monitor = "val_loss", mode = "min", save_best_only = True, save_weights_only = True)
        earlystopping = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 10)
        callback_lists = [modelcheckpoint, earlystopping]

        history = model.fit(train_generator,
                            epochs = epochs,
                            callbacks = callback_lists,
                            validation_data = test_generator)
        
        model.load_weights(checkpoint_path)
        model_path = save_checkpoint_folder + str(fold_var) + ".h5"
        model.save(model_path)

        results = model.evaluate(test_generator)
        results = dict(zip(model.metrics_names, results))

        validation_accuracy.append(results['acc'])
        validation_loss.append(results['loss'])

        keras.backend.clear_session()

        fold_var += 1

        val_acc += results['acc']/5

    print(val_acc)
