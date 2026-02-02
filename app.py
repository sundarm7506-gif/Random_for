import streamlit as st
import pandas as pd
import pickle

st.title("Simple Random Forest App")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset (just for column names)
df = pd.read_excel(r"random_forest_dataset.xlsx")
features = df.columns[:-1]

st.write("Enter values:")

inputs = []
for f in features:
    val = st.number_input(f, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    pred = model.predict([inputs])
    prob = model.predict_proba([inputs])

    st.success("Prediction: " + str(pred[0]))
    st.write("Probability:", prob)
