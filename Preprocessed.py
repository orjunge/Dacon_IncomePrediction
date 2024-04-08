## 데이터 로드
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

## 컬럼 삭제
"""
EDA 결과 Income과 상관관계 지수가 낮거나,
독립변수의 값이 종속변수 Income에 유의미한 폭이 없을 경우 컬럼 삭제하였습니다.
"""
train = train.drop(['ID', 'Gains', 'Dividends', 'Income_Status', 'Race', 'Tax_Status', 
                    'Household_Status', 'Citizenship'], axis=1)


## Age 구간화 후 Label Encoding 진행
"""
Label Encoder를 진행할 때 변환된 값이 순서/크기를 가지게 되어 중요도가 학습될 수 있는 문제가 있지만,
Age 컬럼의 경우 크기를 학습되어도 괜찮다고 판단하여 LabelEncoding을 선택
"""
bins = range(0, 91, 10)
train['Age'] = pd.cut(train['Age'], bins)

label_encoder = LabelEncoder()
train['Age'] = label_encoder.fit_transform(train['Age'])


## Education_Status
"""
학력을 나누어 (고교 재학, 졸업/학사,석사,박사) 두 집단으로 나누어 Income과의 관계를 분석한 결과
학력에 따라 Income에 영향을 많이 미쳐 One-Hot Encoding을 진행
"""
encoder = OneHotEncoder()
education_status = train['Education_Status'].values.reshape(-1, 1)

encoded_data = encoder.fit_transform(education_status)
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.categories_[0])
encoded_df = encoded_df.astype(int) # float -> int 형식으로 변환

train = pd.concat([train, encoded_df], axis=1)
train = train.drop(['Education_Status'], axis=1)


## Gender -> 범주화
train['Gender'] = train['Gender'].map({'M': 1, 'F': 0})


## Employment_Status, Industries_Status, Hispanic_Origin -> Label Encoding
label_encoder = LabelEncoder()
train['Employment_Status'] = label_encoder.fit_transform(train['Employment_Status'])

label_encoder = LabelEncoder()
train['Occupation_Status'] = label_encoder.fit_transform(train['Occupation_Status'])

label_encoder = LabelEncoder()
train['Industry_Status'] = label_encoder.fit_transform(train['Industry_Status'])

label_encoder = LabelEncoder()
train['Hispanic_Origin'] = label_encoder.fit_transform(train['Hispanic_Origin'])

label_encoder = LabelEncoder()
train['Household_Summary'] = label_encoder.fit_transform(train['Household_Summary'])

## 파생변수 생성 = Employment_Status + Occupation_Status
"""
변수의 분포가 동일하여 둘은 값을 더하여 파생변수 값으로 생성
"""
train['Occu+Employ'] = train['Occupation_Status'] + train['Employment_Status']
train.drop(columns=['Occupation_Status', 'Employment_Status'], axis=1, inplace=True)


## Martial_Status
"""
One-Hot 인코딩이 가지는 과대적합 문제가 우려되었찌만 고유값이 3가지 뿐이라 One Hot Enocidng으로 전처리 
"""
encoder = OneHotEncoder()
martial_status = train['Martial_Status'].values.reshape(-1, 1)

encoded_data = encoder.fit_transform(martial_status)
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.categories_[0])

encoded_df = encoded_df.astype(int)

train = pd.concat([train, encoded_df], axis=1)
train = train.drop(['Martial_Status'], axis=1)


## Birth_Country, Father, Mother
"""
빈도가 높은 값을 1, 그 외의 값은 0으로 하여 데이터 단순화
"""
train['US_Born'] = (train['Birth_Country'] == 'US').astype(int)
train['US_Born'] = (train['Birth_Country (Father)'] == 'US').astype(int)
train['US_Born'] = (train['Birth_Country (Mother)'] == 'US').astype(int)

train.drop(columns=['Birth_Country', 'Birth_Country (Father)', 'Birth_Country (Mother)'], axis=1, inplace=True)


## Losses
"""
2만 데이터 중 1만 9천 이상의 데이터가 0의 값을 가지고 있어서
0일 경우 1, 그 외 데이터는 0으로 치환
"""
train['Losses'] = train['Losses'].apply(lambda x: 1 if x == 0 else 0)


## 컬럼 순서 조정
income_column = train.pop('Income')
train['Income'] = income_column
train.head()











