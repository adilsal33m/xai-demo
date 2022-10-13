import streamlit as st

st.title('Explainable AI for Health Sector')
imp = """
### Version History
##### Version 0.1
- [LIME](/LIME) explanation added
- [SHAP](/SHAP) explanation added
##### Version 0.2
- [NICE](/NICE) explanation added
##### Version 0.3
- Support added to test multiple ML models
##### Version 0.4
- Neural Nets added
##### Version 0.5
- [Anchors](/Anchors) explanation added
- [KernelSHAP](/KernelSHAP) explanation added
##### Version 0.6
- Reference papers added for each explanation (can be accessed via Read More link on respective page)
##### Version 0.7
- [DiCE](/DiCE) added (may not work for neural networks)
"""
st.markdown(imp)

todo = """
### Upcoming Features
- ELI5
- Skope Rules
- Prototypes
- More Counterfactual Explanations
- Unified page for all explanations
"""
st.markdown(todo)