from functools import cache
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nice import NICE
from dill import loads

@st.cache(allow_output_mutation=True)
def load_file():
    data = None
    with open('xai-app.dump','rb') as f:
        data = f.read()
        
    return loads(data)

@st.cache(allow_output_mutation=True)
def get_model():
    data = load_file()
    return data['model']

@st.cache(allow_output_mutation=True)
def get_explainer():
    data = load_file()
    X = data['data']['X']
    y = data['data']['y']
    nice_explainer = NICE(X_train = X,
               predict_fn=predict_proba,
               y_train = y,
               cat_feat=[1,3,4,5,6,7,8,9],
               distance_metric='HEOM',
               num_normalization='minmax',
               optimization='sparsity',
               justified_cf=True
               )
    return nice_explainer

def prepare_df():
    data = load_file()
    columns = data['columns']
    df = pd.DataFrame(columns= columns)
    
    row = {
        'BMI': bmi,
        'SleepTime': sleep,
        'Smoking_Yes': 1 if smoking else 0,
        'AlcoholDrinking_Yes': 1 if alcohol else 0,
        'DiffWalking_Yes': 1 if diff_walking else 0,
        'Sex_Male': 1 if sex == 'Male' else 0,
        'Diabetic': 1 if diabetes else 0,
        'PhysicalActivity_Yes': 1 if p_activity else 0,
        'Asthma_Yes': 1 if asthma else 0,
        'KidneyDisease_Yes': 1 if k_disease else 0,
        'AgeCategory': age
    }
    df = df.append(row, ignore_index=True)    
    return  df

def predict(x):
    model = get_model()
    return model.predict_proba(x)[0]

def explain(row):
    explainer = get_explainer()
    exp = explainer.explain(row)
    return exp

def predict_proba(x):
    model = get_model()
    y = model.predict_proba(x)
    return np.apply_along_axis(lambda x: (1,0) if x[1] < float(threshold)/100 else (0,1),1,y)


#Code starts here
model = get_model()
st.title('Health Assessment with Counterfactual Explanations using NICE')

age = st.slider('Age',1,120,value=25)
sex = st.selectbox('Gender',['Male','Female'])
height = st.slider('Height (in cms)',100,250,value=172)
weight = st.slider('Weight (in kgs)',10,200,value=67)
bmi = round(weight/pow(height/100,2),1)
st.write("BMI: {}".format(bmi))
sleep = st.slider('Sleep (in hours)',0,12,value=6)

st.subheader('Habits')
smoking = st.checkbox('Smoking')
alcohol = st.checkbox('Alcohol Consumption')
p_activity = st.checkbox('Physical Activity')

st.subheader('Known Ailments')
asthma = st.checkbox('Asthma')
k_disease = st.checkbox('Kidney Disease')
diff_walking = st.checkbox('Walking Difficulties')
diabetes = st.checkbox('Diabetes')

st.subheader('Model Parameters')
threshold = st.slider('CF Threshold (%)',0,100,value=5)

if st.button('Predict'):
    st.subheader('Prediction')
    row = prepare_df()
    prediction = predict(row)
    
    st.write('Chances of suffering from heart disease:')
    st.markdown('<h3>{}%</h3>'.format(round(prediction[1]*100,1)), unsafe_allow_html=True)
    st.write('Chances of NOT suffering from heart disease:')
    st.markdown('<h3>{}%</h3>'.format(round(prediction[0]*100,1)), unsafe_allow_html=True)
    
    if(prediction[1] > float(threshold)/100):
        st.subheader('Counterfactual Explanation')
        st.write('Original Instance')
        row
        results = explain(row.values)
        prediction = predict(results)
        results = pd.DataFrame(results, columns=row.columns)
        st.write('Counterfactual Instance')
        results
        st.write('Chances of suffering from heart disease:')
        st.markdown('<h3>{}%</h3>'.format(round(prediction[1]*100,1)), unsafe_allow_html=True)
        st.write('Chances of NOT suffering from heart disease:')
        st.markdown('<h3>{}%</h3>'.format(round(prediction[0]*100,1)), unsafe_allow_html=True)