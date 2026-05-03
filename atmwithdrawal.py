import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------- PAGE STYLE ----------------
st.set_page_config(page_title="ATM Prediction", page_icon="🏧", layout="wide")

st.markdown("""
<style>
.stApp {
background: linear-gradient(to right,#4facfe,#00f2fe);
}

.title{
text-align:center;
font-size:40px;
color:white;
font-weight:bold;
}

.box{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 10px rgba(0,0,0,0.2);
}

div.stButton > button{
background:#ff4b4b;
color:white;
font-size:18px;
border-radius:10px;
height:50px;
width:100%;
}

div.stButton > button:hover{
background:#ff0000;
color:white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<p class="title">🏧 ATM Withdrawal Prediction System</p>', unsafe_allow_html=True)

st.write("### Enter details to predict withdrawal amount")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("atm_withdrawal_dataset.csv")
data = data.drop(columns=["transaction_id"])

X = data.drop("withdrawal_amount", axis=1)
y = data["withdrawal_amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- INPUTS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    day = st.number_input("Day of Week (0-6)",0,6)
    weekend = st.selectbox("Is Weekend",[0,1])
    month_start = st.selectbox("Is Month Start",[0,1])
    month_end = st.selectbox("Is Month End",[0,1])

with col2:
    location = st.number_input("Location Type",0,3)
    account_type = st.number_input("Account Type",0,3)
    balance = st.number_input("Account Balance")
    income = st.number_input("Monthly Income")

with col3:
    age = st.number_input("Age",18,80)
    prev = st.number_input("Previous Withdrawals",0,20)
    hour = st.number_input("Hour of Day",0,23)
    shops = st.number_input("Nearby Shops",0,50)

st.write("")

# ---------------- PREDICTION ----------------
if st.button("Predict Withdrawal Amount"):

    input_data = [[
        day, weekend, month_start, month_end,
        location, account_type, balance,
        income, age, prev, hour, shops
    ]]

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Withdrawal Amount: ₹ {prediction[0]:.2f}")

# ---------------- DATASET VIEW ----------------
if st.checkbox("Show Dataset"):
    st.dataframe(data)
