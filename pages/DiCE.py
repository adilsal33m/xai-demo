from functools import cache
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost

import dice_ml
from dill import loads
from keras.models import load_model
import copy

predict_proba = None

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
    global predict_proba
    selected = model_list()[model]
    if selected == 'neural_net':
        return load_model('nn_model.h5')
    else:
        data = load_file()
        m = copy.deepcopy(data['models'][selected])
        predict_proba = m.predict_proba
        dice_predict_proba = lambda x:  np.apply_along_axis(lambda x: (1,0) if x[1] < float(get_threshold())/100 else (0,1),1,predict_proba(x))
        m.predict_proba = dice_predict_proba
        return m 
 
def get_explainer():
    data = load_file()
    X = data['data']['X']
    y = data['data']['y']

    d = dice_ml.Data(dataframe=pd.concat([pd.DataFrame(X,columns=data['columns']).astype(int),
                                          pd.DataFrame(y,columns=['HeartDisease'])],axis=1),
                     continuous_features=list(data['columns']),
                     outcome_name='HeartDisease')
    
    selected = model_list()[model]
    if selected == 'neural_net':
        m = dice_ml.Model(model=get_model(), backend="TF2")
        explainer = dice_ml.Dice(d, m, method="gradient")
        return explainer
    else:
        m = dice_ml.Model(model=get_model(), backend="sklearn")
        explainer = dice_ml.Dice(d, m, method="random")
        return explainer
    

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
    df = df.astype(int)  
    return  df

def predict(x):
    model = get_model() # gets predict_proba as a side effect - BAD Practice
    if predict_proba == None:
        v = 1 - model.predict(x)
        return np.array([[x[0][0],x[1][0]] for x in list(zip(v,1-v))])[0]
    else:
        return predict_proba(x)[0]

def get_threshold():
    return threshold

def explain(row):
    explainer = get_explainer()
    e1 = explainer.generate_counterfactuals(row,
                                  total_CFs=1,
                                  desired_class="opposite",
                                  features_to_vary=["BMI", "SleepTime",'Smoking_Yes',"AlcoholDrinking_Yes",'PhysicalActivity_Yes'])
    return e1


#Code starts here
st.markdown('## Health Assessment with Counterfactual Explanations using DiCE')
st.markdown(r"[Read More](https://arxiv.org/abs/1905.07697.pdf)")

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
model = st.selectbox('Select Model',model_list().keys())

if st.button('Predict'):
    st.subheader('Prediction')
    row = prepare_df()
    prediction = predict(row)
    
    st.write('Chances of suffering from heart disease:')
    st.markdown('<h3>{}%</h3>'.format(round(prediction[1]*100,1)), unsafe_allow_html=True)
    st.write('Chances of NOT suffering from heart disease:')
    st.markdown('<h3>{}%</h3>'.format(round(prediction[0]*100,1)), unsafe_allow_html=True)
    
    try:
        results = explain(row)
        st.write('Original:')
        row
        
        row = results.cf_examples_list[0].final_cfs_df
        del row['HeartDisease']
        st.write('Counterfactual:')
        row
        prediction = predict(row)
        
        st.write('Chances of suffering from heart disease:')
        st.markdown('<h3>{}%</h3>'.format(round(prediction[1]*100,1)), unsafe_allow_html=True)
        st.write('Chances of NOT suffering from heart disease:')
        st.markdown('<h3>{}%</h3>'.format(round(prediction[0]*100,1)), unsafe_allow_html=True)
    except:
        st.write('No counterfactual instance found!')