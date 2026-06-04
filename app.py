
# ==========================================
# STUDENT PERFORMANCE PREDICTOR
# ==========================================

import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("student_data.csv")
data.head()


X = data[["age", "studytime", "G1", "G2"]]
y = data["G3"]

# Train Model 
print(X_train.shape)
print(X_test.shape)

# ADD THESE 3 LINES:
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions[:10])

from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("MAE:", mae)
print("R2 Score:", r2)

print("===== STUDENT PERFORMANCE PREDICTOR =====")

# User Input
age = int(input("Enter Student Age: "))
studytime = int(input("Enter Study Time (1-4): "))
G1 = int(input("Enter G1 Grade (0-20): "))
G2 = int(input("Enter G2 Grade (0-20): "))

# Create Input Data
new_student = pd.DataFrame({
    "age": [age],
    "studytime": [studytime],
    "G1": [G1],
    "G2": [G2]
})

# Predict
predicted_G3 = model.predict(new_student)[0]

print("\n----- RESULT -----")
print(f"Predicted Final Grade (G3): {predicted_G3:.2f}")

# Performance Category
if predicted_G3 >= 16:
    print("Performance: Excellent")
elif predicted_G3 >= 12:
    print("Performance: Good")
elif predicted_G3 >= 8:
    print("Performance: Average")
else:
    print("Performance: Needs Improvement")
