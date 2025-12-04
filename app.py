import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Page Configuration
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

# Custom CSS for a "Dark Mode" Tech feel
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('churn_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ Model files not found! Please make sure 'churn_model.h5' and 'scaler.pkl' are in the same folder.")
    st.stop()

# 2. Header Section
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ ChurnGuard AI")
    st.markdown("### Deep Learning Customer Retention System")
    st.write("This tool uses an **Optimized Neural Network (128 Neurons)** to predict high-risk banking customers with **83%+ Accuracy**.")

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100) # Tech Icon

st.divider()

# 3. Input Section (Sidebar for inputs)
st.sidebar.header("👤 Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geo = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
has_cr_card = st.sidebar.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
is_active = st.sidebar.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0)

# 4. Preprocessing (Exact Match to Training)
# Logic: [CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Germany, Spain, Male]

geo_germany = 1 if geo == "Germany" else 0
geo_spain = 1 if geo == "Spain" else 0
gender_male = 1 if gender == "Male" else 0
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_val = 1 if is_active == "Yes" else 0

input_data = pd.DataFrame([[
    credit_score, age, tenure, balance, num_products, has_cr_card_val, is_active_val, salary, geo_germany, geo_spain, gender_male
]], columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male'])

# Scale Input
input_scaled = scaler.transform(input_data)

# 5. Prediction
if st.sidebar.button("🔍 Run Risk Analysis", type="primary"):
    prediction_prob = model.predict(input_scaled)[0][0]
    
    # Layout for Results
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Risk Assessment")
        if prediction_prob > 0.5:
            st.error(f"⚠️ **HIGH RISK ALERT**")
            st.markdown(f"<h1 style='color: #ff4b4b;'>{prediction_prob*100:.1f}%</h1>", unsafe_allow_html=True)
            st.write("This customer has a high probability of leaving.")
        else:
            st.success(f"✅ **SAFE / RETAINED**")
            st.markdown(f"<h1 style='color: #09ab3b;'>{prediction_prob*100:.1f}%</h1>", unsafe_allow_html=True)
            st.write("This customer is stable.")

    with res_col2:
        st.subheader("Model Confidence")
        st.progress(int(prediction_prob * 100))
        st.caption("Neural Network Confidence Score")
        
        # Actionable Insight
        st.subheader("Recommended Action")
        if prediction_prob > 0.7:
            st.warning("👉 **Immediate Action:** Offer Tenure Bonus or Lower Interest Rate.")
        elif prediction_prob > 0.5:
            st.info("👉 **Watchlist:** Send personalized engagement email.")
        else:
            st.write("👉 No immediate action required.")