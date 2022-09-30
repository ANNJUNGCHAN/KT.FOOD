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

def train(flow_dataset,checkpoint_folder) :
    """
    모델 학습을 위한 함수
    - 5KFOLD로 학습한 후, accuracy의 평균을 출력
    ---input---
    flow_dataset : 이미지 데이터 경로와 target으로 이루어진 데이터셋(dir,target)
    checkpoint_folder : 모델을 저장할 경로(끝에 /를 붙인다)
    ---output---
    5KFOLD의 accuracy 평균
    """
    validation_accuracy = []
    validation_loss = []
    val_acc = 0

    y = flow_dataset["target"]
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)


    fold_var = 1 # 모델 저장용

    for train_index, val_index in skf.split(flow_dataset, flow_dataset["target"]) :

        train_data = flow_dataset.iloc[train_index]
        validation_data = flow_dataset.iloc[val_index]

        # reset index for prevent error
        train_data.reset_index(drop = True, inplace = True)

        # Load images using Keras ImageDataGenerator

        # Train Generator
        datagen_train = ImageDataGenerator(rescale=1./255,
                                        rotation_range=30,
                                            shear_range=0.2,
                                            zoom_range=0.4,
                                            horizontal_flip=True)
        
        # Flow from dataframe
        train_generator = datagen_train.flow_from_dataframe(
            dataframe=train_data,
            x_col='dir',
            y_col='target',
            batch_size=64,
            seed=7,
            shuffle=True,
            class_mode='categorical',
            target_size=(224,224),
        )
            
        # Test Generator
        datagen_test = ImageDataGenerator(rescale=1./255)

        # Flow from dataframe
        test_generator = datagen_test.flow_from_dataframe(
            dataframe=validation_data,
            x_col='dir',
            y_col='target',
            batch_size=64,
            seed=7,
            shuffle=True,
            class_mode='categorical',
            target_size=(224,224),
        )
        
        ### Model ###
        keras.backend.clear_session()

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

        ###

        ### MODEL DEFINE ###
        model = Model(input_train, dense4)
        ###

        ### MODEL COMPILE ###
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])
        ###

        ### MODEL FIT ###
        checkpoint_path = checkpoint_folder + str(fold_var) + ".ckpt"
        modelcheckpoint = ModelCheckpoint(checkpoint_path, monitor = "val_loss", mode = "min", save_best_only = True, save_weights_only = True)
        earlystopping = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 100)
        callback_lists = [modelcheckpoint, earlystopping]

        history = model.fit(train_generator,
                            epochs = 200,
                            callbacks = callback_lists,
                            validation_data = test_generator)
        ###

        model_path = checkpoint_folder + str(fold_var) + ".h5"
        model.save(model_path)

        results = model.evaluate(test_generator)
        results = dict(zip(model.metrics_names, results))

        validation_accuracy.append(results['accuracy'])
        validation_loss.append(results['loss'])

        keras.backend.clear_session()

        fold_var += 1

        val_acc += results['accuracy']/5

    print(val_acc)