## 라이브러리 임포트
import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder


## 데이터로드
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

## 데이터 확인
train.info()
train.isnull().sum() # 결측치 없음
train.describe()
train.Education_Status.value_counts() # Education 컬럼 값 파악


## EDA
### 이산형, 연속형 DF 생성
tr_int = train[train.describe().columns]
tr_str = train[train.columns.difference(train.describe().columns)]


### 연속형의 최빈값 확인
plt.figure(figsize=(16, 6))

features = train_int.columns.values
unique_max_train = []
unique_max_test = []

for feature in features:
    values = train[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = test[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])

np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).\
             sort_values(by='Max duplicates', ascending=False).head())



### 상관관계 분석 -> Working_Week(Yearly) > Age > Losses > Gains > Dividends 컬럼 순서로 종속변수 Income과 가장 관계가 높음
corr_df = tr_int.copy()

corr_matrix = corr_df.corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(corr_matrix, mask=mask, annot=True, annot_kws={"size":8})
plt.suptitle("Correlation(Numeric)")
plt.show()


### Age-Income => 20~40 세대가 가장 높은 소득을 얻고, 이후부턴 감소함
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].scatter(tr_int['Age'], tr_int['Income'], s=20, edgecolor='black', linewidth=0.3, alpha=0.5, color='pink')
axes[0].set_title('Scatterplot, Age-Income')

tr_int['Age'].plot(kind = 'box',ax=axes[1])
axes[1].set_title('Boxplot, Age')

tr_int['Age'].hist(bins =  50,ax=axes[2], color='pink', edgecolor='black')
axes[2].set_title('Histogram, Age')

plt.tight_layout()
plt.show()


### Working_Week( (Yearly)-Income
"""
최대값과 중앙값이 일치하며, 이상치 없음
값이 양 극단으로 치우쳐 있음 0 또는 52 -> 범주형으로 변환하는 것 고려하기
"""
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].scatter(tr_int['Working_Week (Yearly)'], tr_int['Income'], s=20, edgecolor='black', linewidth=0.3, alpha=0.5, color='pink')
axes[0].set_title('Scatterplot, Working_Week (Yearly)-Income')

tr_int['Working_Week (Yearly)'].plot(kind = 'box',ax=axes[1])
axes[1].set_title('Boxplot, Working_Week (Yearly)')

tr_int['Working_Week (Yearly)'].hist(bins =  50,ax=axes[2], color='pink', edgecolor='black')
axes[2].set_title('Histogram, Working_Week (Yearly)')

plt.tight_layout()
plt.show()


### Gains-Income
"""
값이 0에 몰려 있고,
이상치 많음
Gains 값이 0일수록 고소득자가 많이 분포되어 있음
"""
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].scatter(tr_int['Gains'], tr_int['Income'], s=20, edgecolor='black', linewidth=0.3, alpha=0.5, color='pink')
axes[0].set_title('Scatterplot, Gains')

tr_int['Gains'].plot(kind = 'box',ax=axes[1])
axes[1].set_title('Boxplot, Gains')

tr_int['Gains'].hist(bins =  50,ax=axes[2], color='pink', edgecolor='black')
axes[2].set_title('Histogram, Gains')

plt.tight_layout()
plt.show()


### Losses-Income
"""
Gains와 마찬가지로 0의 값이 몰려있으며,
이상치가 많음
Losses 값이 0일수록 고소득자가 많이 분포되어 있음
"""
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].scatter(tr_int['Losses'], tr_int['Income'], s=20, edgecolor='black', linewidth=0.3, alpha=0.5, color='pink')
axes[0].set_title('Scatterplot, Losses')

tr_int['Losses'].plot(kind = 'box',ax=axes[1])
axes[1].set_title('Boxplot, Losses')

tr_int['Losses'].hist(bins =  50,ax=axes[2], color='pink', edgecolor='black')
axes[2].set_title('Histogram, Losses')

plt.tight_layout()
plt.show()


### 종속변수 Income
"""
피쳐 특성 상 이상치 많으며,
소득이 0인 사람이 많이 분포되어 있음
"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

tr_int['Income'].plot(kind = 'box',ax=axes[0])
axes[0].set_title('Boxplot, Income')

tr_int['Income'].hist(bins =  50,ax=axes[1], color='pink', edgecolor='black')
axes[1].set_title('Histogram, Income')

plt.tight_layout()
plt.show()


### 연속형 변수의 로그 변환 진행 => 이상치 완화하여 정규성 가진 분포 형태로 바꾸기 위함
"""
진행 컬럼: Age, Gains, Dividends, Income, Losses, Working_Week (Yearly)
로그변환 진행했으나 Age 컬럼을 제외하고 극단 값을 갖는 피쳐였음
"""
np.log(tr_int+1).hist(bins=100)
plt.tight_layout()
plt.show()


### 개별 컬럼 분석
"""
아래와 같은 형식으로 개별 컬럼을 분석하여 전처리 방식을 결정하였습니다.
분포 수와 소득이 정비례 또는 반비례 하는 경우 하는 경우 -> Label 또는 One-hot Encoding 을 진행하여 수치화

Age 컬럼과 같이 0에서 90까지 다양한 값으로 분포한 수치형은 구간화로 전처리,

Gender 컬럼과 같이 M 또는 F 값을 갖는 컬럼은 범주화,

Birth_Country 컬럼 처럼 하나의 국가에 값이 90% 가까이 몰려있는 경우 -> 빈도가 높은 US 국가는 1, 그 외 국가는 0으로 치환

학력과 직업의 분포가 비슷하여 이 둘은 Label Encoding을 진행한 후 값을 더하여 새로운 파생변수로 만듦
"""
result = train['Employment_Status'].value_counts()
print(result)


mean_income_by_category = train.groupby('Employment_Status')['Income'].mean()
mean_income_by_category = mean_income_by_category.sort_values(ascending=False)

plt.figure(figsize=(6, 10))
train['Employment_Status'].value_counts().plot(marker='o', linestyle='', color='blue', label='Counts')
mean_income_by_category.plot(kind='bar', color='orange', alpha=0.7, label='Mean Income')

plt.title('Employment_Status Counts and Mean Income')
plt.xlabel('Employment_Status')
plt.ylabel('Counts / Mean Income')

plt.legend()
plt.grid(True)
plt.show()
