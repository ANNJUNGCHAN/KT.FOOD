# 예선모델
![슬라이드1](https://user-images.githubusercontent.com/89781598/193318478-ab62900a-1511-41e4-bc25-91acb6896a8e.JPG)

### 개요
![슬라이드2](https://user-images.githubusercontent.com/89781598/193318547-f69aa137-2fce-484f-8915-3e179ba53972.JPG)

### 파일 경로
```
📦QualifyingRound
 ┣ 📜continue_train.py
 ┣ 📜DataAugmentation.py
 ┣ 📜predict.py
 ┣ 📜train.py
 ┗ 📜__init__.py
```
### 파일
- QualifyRound : 예선에서 사용했던 코드들이 담겨져 있습니다.
    - continue_train.py : train.py를 이어서 학습하기 위한 코드
    - DataAugmentation.py : Background Remove를 통해 이미지의 배경을 지워주기 위한 코드와 경로와 Target에 대한 정보를 가지고 있는 데이터 프레임을 생성하는 함수
    - predict.py : 학습된 모델에서 예측을 하기 위한 코드
    - train.py : 모델을 학습시키기 위한 코드
    
### Remove Background
![슬라이드3](https://user-images.githubusercontent.com/89781598/193321049-9e9f4db0-1c29-410b-992e-dedfa53e4f28.JPG)

### 모델 설명
![슬라이드4](https://user-images.githubusercontent.com/89781598/193321084-504d4e7d-97d5-4a22-adec-e7a47a5330ea.JPG)
![슬라이드5](https://user-images.githubusercontent.com/89781598/193321089-ab63417b-a710-44c6-a50d-16cbcbb9d1c6.JPG)
![슬라이드6](https://user-images.githubusercontent.com/89781598/193321092-89360dde-1766-4e59-b3d9-5cdde9373cc3.JPG)
![슬라이드7](https://user-images.githubusercontent.com/89781598/193321094-eb510c72-a16d-47ea-adb2-b878cb3ab18a.JPG)
![슬라이드8](https://user-images.githubusercontent.com/89781598/193321095-5955f973-6159-4f8a-9251-a2d29aa5198c.JPG)
![슬라이드9](https://user-images.githubusercontent.com/89781598/193321096-8bf95ab5-b213-48fd-b7af-f49a5b11df86.JPG)
![슬라이드10](https://user-images.githubusercontent.com/89781598/193321098-da30f5ab-d47e-42e1-aa1b-c48400ddc59f.JPG)
![슬라이드11](https://user-images.githubusercontent.com/89781598/193321102-eead7132-badc-41c6-921a-405e60c80ac6.JPG)
![슬라이드12](https://user-images.githubusercontent.com/89781598/193321105-1b8b2be4-0780-4829-93b3-f5172ce87648.JPG)

### 새로운 이미지 예측하기!
#### 들어가기
- 아래와 같은 음식들을 맞출 수 있습니다.(50개)
```
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
```
#### 환경설정
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

#### 예측하기
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
