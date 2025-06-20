import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("rf_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("💼 Bank Campaign Subscription Predictor")
st.markdown("Fill in customer info below to predict whether they'll subscribe to a term deposit.")

# Input form
age = st.slider("Age", 18, 95, 30)
job = st.selectbox("Job", ['admin.', 'technician', 'blue-collar', 'management', 'retired', 'services', 'entrepreneur', 'unemployed', 'student', 'housemaid', 'self-employed', 'unknown'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox("Has Credit in Default?", ['yes', 'no'])
balance = st.number_input("Account Balance", -5000, 100000, 500)
housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
loan = st.selectbox("Has Personal Loan?", ['yes', 'no'])
contact = st.selectbox("Contact Method", ['cellular', 'telephone', 'unknown'])
day = st.slider("Day of Month Contacted", 1, 31, 15)
month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
campaign = st.slider("Number of Contacts This Campaign", 1, 50, 1)
pdays = st.slider("Days Since Last Contact", -1, 999, -1)
previous = st.slider("Number of Previous Contacts", 0, 20, 0)
poutcome = st.selectbox("Previous Campaign Outcome", ['success', 'failure', 'other', 'unknown'])

# Form into DataFrame
input_df = pd.DataFrame([{
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'balance': balance,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'day': day,
    'month': month,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome
}])

# Prediction
if st.button("Predict"):
    input_encoded = encoder.transform(input_df)
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result")
    if pred == 1:
        st.success(f"✅ Customer is likely to subscribe (Probability: {prob:.2%})")
    else:
        st.error(f"❌ Customer is not likely to subscribe (Probability: {prob:.2%})")
