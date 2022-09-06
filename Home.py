import streamlit as st

st.title('Explainable AI for Health Sector')
st.subheader('Implemented')
imp = """- [x] [LIME](/LIME)
"""
st.markdown(imp)

st.subheader('Upcoming Features')
todo = """
- [ ] SHAP Values
- [ ] Counterfactual Explanations
- [ ] Ability to select ML models  
"""
st.markdown(todo)