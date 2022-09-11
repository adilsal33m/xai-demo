import streamlit as st

st.title('Explainable AI for Health Sector')
st.subheader('Implemented')
imp = """
- [x] [LIME](/LIME)
- [x] [SHAP](/SHAP)
- [x] [NICE](/NICE)
"""
st.markdown(imp)

st.subheader('Upcoming Features')
todo = """
- [ ] Counterfactual Explanations
- [ ] Ability to select ML models  
"""
st.markdown(todo)