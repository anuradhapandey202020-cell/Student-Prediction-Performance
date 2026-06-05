import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.title("Student Performance Prediction")

df = pd.read_csv("dataset.csv")

X = df[["study_hours", "attendance", "previous_score"]]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

study_hours = st.number_input("Study Hours")
attendance = st.number_input("Attendance")
previous_score = st.number_input("Previous Score")

if st.button("Predict"):
    result = model.predict([[study_hours, attendance, previous_score]])

    if result[0] == 1:
        st.success("PASS")
    else:
        st.error("FAIL")