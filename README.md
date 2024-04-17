# Dacon_IncomePrediction
다항 릿지 회귀 모델, OLS, LinearRegression, RandomForest를 활용하여 [소득 예측 AI 해커톤 대회안내 링크](https://dacon.io/competitions/official/236230/overview/description)을 진행했던 코드입니다.
- README.md 하단에서 Streamlit 구현한 화면을 확인하실 수 있습니다.

## 파일 설명
- 1_EDA : EDA 진행하여 전처리 사항을 결정
- 2_Preprocessed : 전처리 과정
- 3_Models
  - Statsmodels의 OLS를 사용하여 독립, 종속 변수간 통계적 유의성을 확인
  - sk-learn의 Pipeline 클래스로 교차검증지수 기반으로 모델 선택
  - Decision Tree, RandomForest, LinearRegression으로 모델링 수행, Ridge 규제와 Scaling, GridSearch로 성능 개선 시도
- 4_최종모델-다항릿지회귀.ipynb
  - High Cardinality를 가진 컬럼 전처리에 효과적인 Target Encoder를 추가하여 전처리를 다시 진행한 후,
  - 비선형 데이터 모델링에 강한 다항 모델을 시도했고, L2(Ridge) 규제를 적용하여 R^2 값을 0.28까지 끌어올려 최종 모델로 결정했습니다.
  
---
## 대회요약
### 대회 주제
개인 특성 데이터를 활용하여 개인 소득 수준을 예측하는 AI 모델 개발
### 실제 참여 기간
2024.04.02 ~ 2024.04.08 (약 1주일)

### 최종순위
410등/1,164명 (Private 순위)  |  최종점수: 588.51337

# Streamlit App
- 사용자 정보 값을 받아 소득이 예측되는 사용자 화면을 구현하였습니다.
- 앱 로딩에 오랜 시간이 소요되어 캡쳐로 대신합니다.
<img width="961" alt="Streamlit Operating Capture" src="https://github.com/orjunge/Dacon_IncomePrediction/assets/127750133/ba2fe734-7c78-430d-a532-5ad347f44c79">

# 회고
1. LinearRegression, DecisionTree, RandomForest, 다항회귀모델 모델링을 하였고, 그 과정에서 L2(Lasso) 규제, 하이퍼파라미터 최적화 값을 찾는 GridSearch를 진행하였으나 제출 당시에는 작업환경 코랩의 작용이 멈추어 개선된 모델로 예측을 진행하지 못해 상관계수R^2 값이 0.19에 그친 LinearRegression 모델로 제출하였습니다.
1. 정규화, 표준화, GridSearch, 규제 적용 등 여러 방면으로 성능 개선을 시도해 최종적으로 끌어올린 성능 지표 R^2 값이 0.28에 그친 점이 아쉬웠습니다.
1. OLS로 변수간의 설명도로 변수를 제거하고, 성능을 올리기 위해 규제, 표준화, GridSearch의 adams 파라미터의 역할을 학습할 수 있었습니다.
