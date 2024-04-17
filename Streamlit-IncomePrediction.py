import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from category_encoders import TargetEncoder
import joblib


# 모델 로딩
try:
    model = joblib.load('polynominal_model.pkl')
except Exception as e:
    st.error(f"모델 로딩 중 오류 발생: {e}")
    st.stop()
    
    
# 사이드바
st.sidebar.title("About")
st.sidebar.write("이 페이지는 DACON에서 진행한 [소득 예측 해커톤 대회]('https://dacon.io/competitions/official/236230/overview/description')에 약 7일간 참가하여 다항릿지회귀 모델링한 결과물입니다.")


st.sidebar.title("Contents")
menu = st.sidebar.radio("원하시는 항목을 클릭해주세요", ['🏠 Introduction', '📊 EDA & Preprocessing Result', '✨ Do it!'])

def preprocess(input_data):
    # 데이터 전처리 함수
    input_data['Gender'] = input_data['Gender'].map({'M': 1, 'F': 0})
    input_data['Working_Week (Yearly)'] = np.where(input_data['Working_Week (Yearly)'] == '<= 40', 0, 1)
    le = LabelEncoder()
    input_data['Martial_Status'] = le.fit_transform(input_data['Martial_Status'])
    scaler = StandardScaler()
    input_data['Age'] = scaler.fit_transform(input_data[['Age']])
    te = TargetEncoder()
    input_data['Industry_Status'] = te.fit_transform(input_data['Industry_Status'], [555])
    input_data['Occupation_Status'] = te.fit_transform(input_data['Occupation_Status'], [555])
    input_data['Birth_Country'] = te.fit_transform(input_data['Birth_Country'], [555])
    input_data['IndustryOccupation'] = input_data['Industry_Status'] + input_data['Occupation_Status']
    return input_data



if menu == '🏠 Introduction':
    st.header("다항릿지회귀 모델을 활용한 소득 예측")
    st.subheader("")
    st.subheader("📊 EDA & Preprocessing Result")
    st.write('데이터의 특징과, 모델을 선정한 이유를 확인하실 수 있습니다.')
    train = pd.read_csv("train_preprocessed_strealit.csv")
    st.write("- 학습 데이터 모양:", train.shape)
    st.table(train.head(2))    
    
    

    st.subheader("")
    st.subheader("✨ Do it!")
    st.write("✨Do it! 탭에서는 몇 가지 고객 정보를 입력한 후 고객님의 예상 소득을 예상할 수 있습니다. \n\n 지금부터 여러분은 고객의 소득에 따라 대출승인을 해야하는 담당자🧑🏻‍💻입니다. \n\n 고객의 대출 승인 여부를 빠르게 판단해야 상황에서 이 페이지가 도움이 되길 바랍니다 :-) \n\n")
    st.subheader(" ")
    

    st.subheader(" ")
    st.subheader(" ")
    st.subheader(" ")
    st.subheader(" ")
    st.markdown("""
        <style>
        .right-gray-italic-text {
            text-align: right;
            color: gray;
            font-style: italic;  /* 글씨를 기울임 */
        }
        </style>
        <div class="right-gray-italic-text">
            last updated. 2024. 04.
        </div>
        """, unsafe_allow_html=True)
    


elif menu == '📊 EDA & Preprocessing Result':

    st.header("🔻Correlation")
    # on = st.toggle('지표간 상관관계', value=True)
    on = st.checkbox('지표간 상관관계', value=True)

    if on:  
        st.write("1. Income과 상관이 높은 상위 피처는 Working_Week(Yearly)와 Age 입니다. \n\n 1. IndustryOccupation 변수는 파생변수로 Income과 양의 상관관계를 가지는 Industry Status 지표와 Occupation Status 지표를 더하여 만들었습니다. ")
        train = pd.read_csv("train_preprocessed_strealit.csv")
        tr_int = train.select_dtypes(include=[np.number])  # 수치형 데이터만 선택
        corr_matrix = tr_int.corr()
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True

        plt.figure(figsize=(8, 4))
        sns.heatmap(corr_matrix, mask=mask, annot=True, annot_kws={"size": 8})
        plt.title("Correlation Matrix(Preprocessed Dataset)")
        st.pyplot(plt)   
    
    
    st.subheader("🔻Distribution of Dataset")
    # on = st.toggle('분포도', value=True)
    on = st.checkbox('분포도', value=True)
    
    if on:
        st.write('1. RandomForest, LinearRegression, DecisionTree, Polynominal 선형회귀 중 비선형성 데이터를 잘 처리하는 Polynominal에 Ridge를 결합하여 모델링하였습니다.')
        train = pd.read_csv("train_preprocessed_strealit.csv")
        
        def features_distribution (df, features):
            i = 0
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(3, 3, figsize=(20, 10))

            for feature in features:
                i+= 1
                plt.subplot(3, 3, i)

                sns.kdeplot(data=train, x=feature, fill=True)
                plt.title(f"Distribution of {feature}", fontsize=8)
                plt.xlabel(feature, fontsize=7)
                locs, labels = plt.xticks()
                plt.tick_params(axis='x', which='major', labelsize=4, pad=-6)

            plt.tight_layout()
            st.pyplot(plt)


        features = train.columns.values[:]
        print(features)

        features_distribution(df=train, features=features)
        
        


elif menu == '✨ Do it!':
    st.title('Income Prediction Model')
    with st.form("my_form"):
        age = st.slider('나이(Age)를 입력하세요', 18, 100, 25)
        gender = st.selectbox('고객님의 성별은 무엇입니까?', ['M', 'F'])
        working_week = st.selectbox('Working Week (Yearly)', ['<= 40', '> 40'])
        industry_status = st.selectbox('Industry Status', ['Retail', 'Tech', 'Medical (except Hospitals)', 'Health', 'Social Services', 'Education'])
        occupation_status = st.selectbox('Occupation Status', ['Technical', 'Services', 'Management'])
        martial_status = st.selectbox('Marital Status', ['Married', 'Single', 'Divorced'])
        birth_country = st.selectbox('Birth Country', ['South Korea', 'US', 'Canada', 'Mexico'])
        submitted = st.form_submit_button("예측하기")
        
        if submitted:
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Working_Week (Yearly)': [working_week],
                'Industry_Status': [industry_status],
                'Occupation_Status': [occupation_status],
                'Martial_Status': [martial_status],
                'Birth_Country': [birth_country]
            })
            processed_data = preprocess(input_data)
            prediction = model.predict(processed_data)
            st.write(f'Predicted Income (year): ${prediction[0]:,.2f}')
            
            
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title("Completed by")
st.sidebar.write("@orjunge who wants to be with data.")
# st.sidebar.write("📨 ahjungyoon@gmail.com")
st.sidebar.write("🔗[Github]('https://github.com/orjunge?tab=repositories) 🔗[LinkedIn]('https://www.linkedin.com/in/yooon/)")
# st.sidebar.write("🔗[LinkedIn]('https://www.linkedin.com/in/yooon/)")