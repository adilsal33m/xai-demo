import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import shap
from dill import loads
from keras.models import load_model

shap.initjs()


@st.cache(allow_output_mutation=True)
def load_file():
    data = None
    with open('xai-app.dump','rb') as f:
        data = f.read()
        
    return loads(data)

def model_list():
    return {
        'Decision Tree': 'dt',
        'Gradient Boosting': 'gradient_boosting',
        'K-Nearest Neighbours': 'knn',
        'Logistic Regression': 'lr',
        'Naive Bayes': 'naive_bayes',
        'Neural Net': 'neural_net',
        'Random Forests': 'random_forest',
        'XGBoost': 'xgboost',
    }

def get_model():
    selected = model_list()[model]
    if selected == 'neural_net':
        return load_model('nn_model.h5')
    else:
        data = load_file()
        return data['models'][selected]
 
@st.cache(allow_output_mutation=True)   
def get_explainer():
    data = load_file()['data']['X']
    samples = data[np.random.choice(data.shape[0], 1000, replace=False)]
    background = shap.maskers.Independent(samples)
    shap_explainer = shap.Explainer(shap_predict, background)
    return shap_explainer

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
    df = df.astype(float)     
    return  df

def predict_proba(x):
    model = get_model()
    try:
        return model.predict_proba(x)
    except AttributeError:
        v = 1 - model.predict(x)
        return np.array([[x[0][0],x[1][0]] for x in list(zip(v,1-v))])

def predict(x):
    return predict_proba(x)[0]

def shap_predict(x):
    m = get_model()
    selected = model_list()[model]   
    if selected == 'neural_net': 
        return np.apply_along_axis(lambda x: x[0], 1, m.predict(x))
    else:
        return m.predict(x)

def explain(row):
    explainer = get_explainer()
    row = row.astype('float')
    shap_values = explainer(row)
    return shap_values


#Code starts here

st.markdown('## Health Assessment with Feature Influence using SHAP')
st.markdown(r"[Read More](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)")

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
model = st.selectbox('Select Model',model_list().keys())

if st.button('Predict'):
    st.subheader('Prediction')
    row = prepare_df()
    prediction = predict(row)
    
    st.write('Chances of suffering from heart disease:')
    st.markdown('<h3>{}%</h3>'.format(round(prediction[1]*100,1)), unsafe_allow_html=True)
    st.write('Chances of NOT suffering from heart disease:')
    st.markdown('<h3>{}%</h3>'.format(round(prediction[0]*100,1)), unsafe_allow_html=True)
    
    st.subheader('Feature influence on prediction')
    
    data = explain(row)
    fig, ax = plt.subplots()
    shap.plots.waterfall(data[0,:],max_display=20)
    st.pyplot(fig)