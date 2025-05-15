import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.pkl")

st.title("🚢 Titanic Survival Prediction App")

st.sidebar.header("Enter Passenger Details")

# User input
pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 30)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 50.0)

# Encode sex
sex_encoded = 1 if sex == "male" else 0

# Prediction
input_features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
prediction = model.predict(input_features)

# Output
if st.button("Predict"):
    if prediction[0] == 1:
        st.success("The passenger would have survived.")
    else:
        st.error("The passenger would not have survived.")
