import streamlit as st

st.title('Explainable AI for Health Sector')
st.subheader('Implemented')
imp = """
- [x] [LIME](/LIME)
- [x] [SHAP](/SHAP)
- [x] [NICE](/NICE)
- [X] Ability to select ML models
- [X] Neural Nets  
- [X] Anchors
- [X] KernelShap Added
"""
st.markdown(imp)

st.subheader('Upcoming Features')
todo = """
- [ ] Prototypes
- [ ] ELI5
- [ ] More Counterfactual Explanations
"""
st.markdown(todo)