# 본선 모델
![슬라이드1](https://user-images.githubusercontent.com/89781598/193316520-6f2a9a46-5e8d-46c2-89eb-665d031b9edf.JPG)

### 들어가기 전에
- 본선에서는 KT에서 제공하는 NGC Pytorch Docker 기반 KT genie Mars 개발환경에서 진행하였습니다.
- 해당 개발 환경에 데이터와 모델을 저장한 파일이 있었지만, 대회 규정상 이를 다운로드하는 행위는 금지되었기 때문에, 현재 깃허브에는 개발에 사용했던 소스코드만 존재합니다.

### 파일 구조

```
📦Final
 ┣ 📜check_score.py
 ┣ 📜make_cutmix.py
 ┣ 📜make_remove_background.py
 ┣ 📜model_1e-4.py
 ┗ 📜model_1e-7_ep20.py
```

### 파일
- Final : 본선에서 사용했던 코드들이 담겨져 있습니다.
    - check_score.py : 최종 모델의 스코어를 체크하기 위한 코드
    - make_cutmix.py : One class Cut Mix 이미지를 생성하기 위한 코드(OCM)
    - make_remove_background.py : Background Removed 이미지를 생성하기 위한 코드(BR)
    - model_1e-4.py : 최초 모델 학습을 위한 코드(learning rate 1e-4로 최초 학습을 진행하였습니다.)
    - model_1e-7_ep20.py : 이어서 모델을 학습하기 위한 코드(model_1e-4.py로 학습했던 모델을 과적합 방지를 위해 learning rate 1e-7로 재학습시켰습니다.)

### 데이터
![슬라이드3](https://user-images.githubusercontent.com/89781598/193317912-cc0b1b1b-bfb8-4824-afd1-ba17e1530553.JPG)
```
- 원래 데이터 200장
- Background Remove 200장
- CutMix 500장
```

### 학습 방법
```
- 1. learning rate 1e-4로 우선적으로 학습
     - train val과 valid val의 간격이 급격하게 벌어진다면, 이 지점의 체크포인트를 저장한다.
- 2. 에서의 체크포인트부터 learning rate 1e-7로 학습하여 최대한 train val과 valid val의 간격이 벌어지지 않게 만든다.
```

### 모델 개형
- VGG-13 모델의 개형을 사용하였습니다.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 256, 256]           1,792
       BatchNorm2d-2         [-1, 64, 256, 256]             128
              ReLU-3         [-1, 64, 256, 256]               0
            Conv2d-4         [-1, 64, 256, 256]          36,928
       BatchNorm2d-5         [-1, 64, 256, 256]             128
              ReLU-6         [-1, 64, 256, 256]               0
         MaxPool2d-7         [-1, 64, 128, 128]               0
            Conv2d-8        [-1, 128, 128, 128]          73,856
       BatchNorm2d-9        [-1, 128, 128, 128]             256
             ReLU-10        [-1, 128, 128, 128]               0
           Conv2d-11        [-1, 128, 128, 128]         147,584
      BatchNorm2d-12        [-1, 128, 128, 128]             256
             ReLU-13        [-1, 128, 128, 128]               0
        MaxPool2d-14          [-1, 128, 64, 64]               0
           Conv2d-15          [-1, 256, 64, 64]         295,168
      BatchNorm2d-16          [-1, 256, 64, 64]             512
             ReLU-17          [-1, 256, 64, 64]               0
           Conv2d-18          [-1, 256, 64, 64]         590,080
      BatchNorm2d-19          [-1, 256, 64, 64]             512
             ReLU-20          [-1, 256, 64, 64]               0
        MaxPool2d-21          [-1, 256, 32, 32]               0
           Conv2d-22          [-1, 512, 32, 32]       1,180,160
      BatchNorm2d-23          [-1, 512, 32, 32]           1,024
             ReLU-24          [-1, 512, 32, 32]               0
           Conv2d-25          [-1, 512, 32, 32]       2,359,808
      BatchNorm2d-26          [-1, 512, 32, 32]           1,024
             ReLU-27          [-1, 512, 32, 32]               0
        MaxPool2d-28          [-1, 512, 16, 16]               0
           Conv2d-29          [-1, 512, 16, 16]       2,359,808
      BatchNorm2d-30          [-1, 512, 16, 16]           1,024
             ReLU-31          [-1, 512, 16, 16]               0
           Conv2d-32          [-1, 512, 16, 16]       2,359,808
      BatchNorm2d-33          [-1, 512, 16, 16]           1,024
             ReLU-34          [-1, 512, 16, 16]               0
        MaxPool2d-35            [-1, 512, 8, 8]               0
AdaptiveAvgPool2d-36            [-1, 512, 7, 7]               0
           Linear-37                 [-1, 4096]     102,764,544
             ReLU-38                 [-1, 4096]               0
          Dropout-39                 [-1, 4096]               0
           Linear-40                 [-1, 4096]      16,781,312
             ReLU-41                 [-1, 4096]               0
          Dropout-42                 [-1, 4096]               0
           Linear-43                  [-1, 100]         409,700
================================================================
Total params: 129,366,436
Trainable params: 129,366,436
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 381.63
Params size (MB): 493.49
Estimated Total Size (MB): 875.87
----------------------------------------------------------------

```
### 데이터 전처리
#### BR : Background Remover 
![슬라이드4](https://user-images.githubusercontent.com/89781598/193316753-d74ab4c7-aebd-40c0-8c19-730624397254.JPG)
#### OCM : One class Cut Mix
![슬라이드5](https://user-images.githubusercontent.com/89781598/193316856-2280428a-0154-423d-81a7-f801c61b666b.JPG)
#### Data Transforms
![슬라이드6](https://user-images.githubusercontent.com/89781598/193316924-1b5ac8af-35be-4ee1-b8a2-0e7310573b71.JPG)

### 모델링
#### VGG-13 is most efficient model
![슬라이드7](https://user-images.githubusercontent.com/89781598/193317774-92f9ce62-b98b-4511-8946-8033a81d262b.JPG)
#### Vison Transformer vs VGG-13
![슬라이드8](https://user-images.githubusercontent.com/89781598/193317818-674f37d9-9f25-4887-ad86-9df167ea08c7.JPG)
#### Scheduled Learning Rate
![슬라이드9](https://user-images.githubusercontent.com/89781598/193317859-b4ca6c66-bbc2-4b81-9900-c7bfbf7d1077.JPG)

### 결과
- 최종 Test 데이터에 대한 스코어는 KT에서 평가하며, 참가자는 알 수 없습니다.

```
Validation Top1 Accuracy : 84.15%
```
