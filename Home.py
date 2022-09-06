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
- [ ] Other ML Models  
"""
st.markdown(todo)