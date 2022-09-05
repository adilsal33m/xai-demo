from functools import cache
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import lime
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
    return data['explainer']

def prepare_df():
    data = load_file()
    columns = data['columns']
    trans_gh = data['transform_gen_health']
    trans_age = data['transform_age_category']
    df = pd.DataFrame(columns= columns)
    
    row = {
        'BMI': bmi,
        # 'PhysicalHealth': p_health,
        # 'MentalHealth': m_health,
        'SleepTime': sleep,
        'Smoking_Yes': 1 if smoking else 0,
        'AlcoholDrinking_Yes': 1 if alcohol else 0,
        # 'Stroke_Yes': 1 if stroke else 0,
        'DiffWalking_Yes': 1 if diff_walking else 0,
        'Sex_Male': 1 if sex == 'Male' else 0,
        'Diabetic_No, borderline diabetes': 1 if diabetes == 'No/Borderline' else 0,
        'Diabetic_Yes': 1 if diabetes == 'Yes' else 0,
        'Diabetic_Yes (during pregnancy)': 1 if diabetes == 'Yes(with Pregnancy)' else 0,
        'PhysicalActivity_Yes': 1 if p_activity else 0,
        'Asthma_Yes': 1 if asthma else 0,
        'KidneyDisease_Yes': 1 if k_disease else 0,
        # 'SkinCancer_Yes': 1 if s_cancer else 0,
        'GenHealth': trans_gh(gen_health),
        'AgeCategory': trans_age(age)
    }
    
    df = df.append(row, ignore_index=True)
    
    return  df

def predict(x):
    model = get_model()
    return model.predict_proba(x)[0]

def explain(row,num_features = 10):
    explainer = get_explainer()
    model = get_model()
    exp = explainer.explain_instance(row,
                                 model.predict_proba,
                                 num_features=num_features)
    
    return exp.as_list()


#Code starts here

st.title('Health Assessment')

age = st.selectbox('Age Category',['18-24','25-29','30-34','35-39','40-44','45-49','50-54',
       '55-59','60-64','65-69','70-74','75-79','80 or older'])
sex = st.selectbox('Gender',['Male','Female'])
height = st.slider('Height (in cms)',100,250)
weight = st.slider('Weight (in kgs)',10,200)
bmi = round(weight/pow(height/100,2),1)
st.write("BMI: {}".format(bmi))
# race = st.selectbox('Race',['Asian','Black','Hispanic','Other','White'])

st.subheader('General Health')
# p_health = st.slider('Physical Health (less indicates better health)',0,30)
# m_health = st.slider('Mental Health (less indicates better health)',0,30)
sleep = st.slider('Sleep (in hours)',0,12)
gen_health = st.selectbox('General Health',['Poor','Fair','Good','Very good','Excellent'])


st.subheader('Habits')
smoking = st.checkbox('Smoking')
alcohol = st.checkbox('Alcohol Consumption')
p_activity = st.checkbox('Physical Activity')

st.subheader('Known Ailments')
asthma = st.checkbox('Asthma')
k_disease = st.checkbox('Kidney Disease')
# s_cancer = st.checkbox('Skin Cancer')
diff_walking = st.checkbox('Walking Difficulties')
# stroke = st.checkbox('Have you suffered from a stroke?')
diabetes = st.selectbox('Diabetes',['No/Borderline','Yes','Yes(with Pregnancy)'])

if st.button('Predict'):
    st.subheader('Prediction')
    row = prepare_df()
    prediction = predict(row)
    
    st.write('Chances of suffering from heart disease:')
    st.markdown('<h3>{}%</h3>'.format(round(prediction[1]*100,1)), unsafe_allow_html=True)
    st.write('Chances of NOT suffering from heart disease:')
    st.markdown('<h3>{}%</h3>'.format(round(prediction[0]*100,1)), unsafe_allow_html=True)
    
    st.subheader('Feature influence on prediction')
    
    data = explain(row.iloc[0].values)
    # for v in data:
    #     color = 'green' if v[1] < 0 else 'red'
    #     st.markdown('<p>{} <span style="color:{}; font-size: 1.5em;">{}</span></p>'.format(v[0],color,abs(round(v[1],3))), unsafe_allow_html=True)
    
    fig, ax = plt.subplots()
    feature = [v[0] for v in data]
    y_pos = np.arange(len(feature))
    performance = [v[1] for v in data]

    ax.barh(y_pos, performance, align='center',color=[('r' if p > 0 else 'g') for p in performance])
    ax.set_yticks(y_pos, labels=feature)
    ax.invert_yaxis()
    ax.set_xlabel('Impact')
    ax.set_title('Red corresponds to Heart Disease')
    st.pyplot(fig)