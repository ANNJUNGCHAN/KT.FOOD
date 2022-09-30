### 사용방법
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
- QRTrainedModel에 가중치가 저장됩니다.
- colab에서 분할압축을 풀어오는 과정에서, 가중치를 복원하지 못하는 에러를 해결할 수 없었습니다.
