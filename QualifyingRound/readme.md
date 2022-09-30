# 새로운 이미지 예측하기!
### 들어가기
- 아래와 같은 음식들을 맞출 수 있습니다.(50개)
```

```
### 환경설정
- 아래의 코드를 먼저 실행시켜주세요!
```
<Colab>
!git clone https://github.com/ANNJUNGCHAN/KT.FOOD.git
!gdown https://drive.google.com/uc?id=1kEs2H7SiFGDn_7cxezP3QlSiC0loJ-q3
!cd /content/KT.FOOD
!unzip -qq "/content/QRTrainedModel.zip"

<Local>
!git clone https://github.com/ANNJUNGCHAN/KT.FOOD.git
!gdown https://drive.google.com/uc?id=1kEs2H7SiFGDn_7cxezP3QlSiC0loJ-q3
!cd "KT.FOOD가 들어있는 경로"
!unzip -qq "QRTrainedModel.zip이 들어있는 경로"
```
- 모델을 이용해 새로운 이미지를 예측하고 싶다면, 위의 코드를 실행하시오.
- QRTrainedModel.zip에 가중치가 저장된 압축파일이 존재합니다.
- colab에서 분할압축을 풀어오는 과정에서, 가중치를 복원하지 못하는 에러를 해결할 수 없었습니다.

### 예측하기
- 아래의 패키지를 먼저 불러와주세요!
```
%cd /content/KT.FOOD/QualifyingRound
import DataAugmentation as DA
import predict
import continue_train
```
- 아래와 같이 폴더가 구성되어 있어야 합니다.
- 이미지는 모두 jpg 파일 형식이어야 합니다.
- 권장되는 이미지 사이즈는 256 by 256 입니다.
```
📦FoodTest
 ┗ 📂Test
 ┃ ┣ 📜test1.jpg
 ┃ ┣ 📜test2.jpg
 ┃ ┣ 📜test3.jpg
 ┃ ┣ 📜test4.jpg
 ┃ ┗ 📜test5.jpg
```
- 예측할 이미지파일의 경로와 예측할 이미지가 무엇인지를 알려주는 target으로 이루어진 데이터 프레임을 만듭니다.
```
df = DA.directory_dataframe("FoodTest의 경로",env = 환경)
```
- 모델을 빌딩합니다.
```
model = continue_train.create_model()
```
- 예측을 진행합니다.(예측한 결과를 벡터로 반환합니다.)
```
pred = predict.predict(model,df,"가중치의 경로")
```
- 실제 예측한 결과를 class의 이름으로 불러옵니다.
```
results = predict.result_predict(pred)
```

