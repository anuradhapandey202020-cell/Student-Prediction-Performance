 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------
# 1. LOAD DATASET
# -------------------------
df = pd.read_csv("dataset.csv")

print("\nDataset Preview:")
print(df.head())

# -------------------------
# 2. SPLIT DATA
# -------------------------
X = df[["study_hours", "attendance", "previous_score"]]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3. TRAIN MODEL
# -------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------
# 4. PREDICTION
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# 5. ACCURACY
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# -------------------------
# 6. CUSTOM PREDICTION
# -------------------------
print("\n--- Student Pass/Fail Prediction ---")
study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance (%): "))
previous_score = float(input("Enter Previous Score: "))

result = model.predict([[study_hours, attendance, previous_score]])

if result[0] == 1:
    print("Prediction: PASS ✅")
else:
    print("Prediction: FAIL ❌")