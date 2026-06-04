import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.title("🎓 Student Performance Predictor")

# Load data
data = pd.read_csv("student_data.csv")

# Train model
X = data[["age", "studytime", "G1", "G2"]]
y = data["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# User Input
st.subheader("Enter Student Details")
age = st.number_input("Age", min_value=10, max_value=25, value=16)
studytime = st.slider("Study Time (1-4)", 1, 4, 2)
G1 = st.number_input("G1 Grade (0-20)", min_value=0, max_value=20, value=10)
G2 = st.number_input("G2 Grade (0-20)", min_value=0, max_value=20, value=10)

if st.button("Predict"):
    new_student = pd.DataFrame({
        "age": [age], "studytime": [studytime],
        "G1": [G1], "G2": [G2]
    })
    predicted_G3 = model.predict(new_student)[0]
    st.success(f"Predicted Final Grade (G3): {predicted_G3:.2f}")

    if predicted_G3 >= 16:
        st.info("Performance: Excellent 🌟")
    elif predicted_G3 >= 12:
        st.info("Performance: Good 👍")
    elif predicted_G3 >= 8:
        st.info("Performance: Average 📊")
    else:
        st.warning("Performance: Needs Improvement 📚")
