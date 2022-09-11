import streamlit as st

st.title('Explainable AI for Health Sector')
st.subheader('Implemented')
imp = """
- [x] [LIME](/LIME)
- [x] [SHAP](/SHAP)
- [x] [NICE](/NICE)
- [ ] Ability to select ML models  
"""
st.markdown(imp)

st.subheader('Upcoming Features')
todo = """
- [ ] Anchors
- [ ] Prototypes
- [ ] ELI5
- [ ] More Counterfactual Explanations
"""
st.markdown(todo)