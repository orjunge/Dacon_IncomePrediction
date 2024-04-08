# Dacon_IncomePrediction
Linear Regression과 Statsmodels의 OLS를 활용하여 소득 예측 모델링을 진행했던 코드입니다.
- [소득 예측 AI 해커톤 대회안내 링크](https://dacon.io/competitions/official/236230/overview/description)

# 파일 설명
- 1_EDA : EDA 진행하여 전처리 사항을 결정
- 2_Preprocessed : 전처리 과정
- 3_Models
  - Statsmodels의 OLS를 사용하여 독립, 종속 변수간 통계적 유의성을 확인
  - sk-learn의 Pipeline 클래스로 교차검증지수 기반으로 모델 선택
  - 위 내용 바탕으로 Decision Tree, RandomForest, LinearRegression으로 모델링 수행, GridSearch로 성능 개선 시도

---
# 대회요약

## 실제 참여 진행 기간
2024.04.06 ~ 2024.04.08

## 최종순위
410등/1,164명 (Private 순위)  |  최종점수: 588.51337

## 데이터 설명
- 원본: train (20000, 23), test(10000, 22)
- 전처리 후: train (20000, 34), test (10000, 33)

## 아쉬운 점
1. LinearRegression과 RandomForest 모델 성능 개선을 위해 L2(Lasso) 규제, 하이퍼파라미터 최적화 값을 찾는 GridSearch를 진행하였으나  제출 당시에는 작업환경 코랩의 작용이 멈추어 개선된 모델로 예측을 진행하지 못해 상관계수R^2 값이 0.19에 그친 LinearRegression 모델로 제출하였습니다. (성능 개선 후 Random Forest 모델의 R^2 값이 0.24까지 올라옴)
1. 상관계수 값이 성능 개선을 시도해도 0.24에 그친 점이 아쉬웠습니다.
