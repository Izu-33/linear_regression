import streamlit as st
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression


model = joblib.load('model.joblib')

st.markdown('# E-commerce Expense Prediction App')
st.markdown('---')
col1, col2 = st.columns(2)

with col1:
    sess = st.number_input('Average session length (in Minutes)')
    app_time = st.number_input('Time on app (in Minutes)')

with col2:
    web_time = st.number_input('Time on website (in Minutes)')
    mem_length = st.number_input('Length of membership (in Months)')

if st.button("Predict"):
    sample = np.array([sess, app_time, web_time, mem_length]).reshape(1, -1)
    prediction = model.predict(sample)[0]
    prediction = f'${prediction:.2f}'
    st.info(f'This customer is likely to spend up to {prediction}')