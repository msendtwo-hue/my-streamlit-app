import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load model
MODEL_PATH = Path("iris_model.pkl")
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Demo Iris", layout="centered")
st.title("Prediksi Spesies Bunga Iris")

# Nama spesies
species_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

model = None
if MODEL_PATH.exists():
    model = load_model()

# Input manual
st.header("Input Manual")
sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Prediksi Manual"):
    if model:
        df = pd.DataFrame([[sl, sw, pl, pw]], columns=["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"])
        pred = model.predict(df)
        st.success(f"Prediksi Spesies: {species_names[pred[0]]}")
    else:
        st.error("Model belum tersedia.")

# Prediksi batch dari CSV
st.header("Prediksi Batch")
uploaded = st.file_uploader("Unggah CSV (format kolom sesuai data iris)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    if st.button("Prediksi Batch"):
        if model:
            preds = model.predict(df)
            # Mapping angka ke nama spesies
            pred_species = [species_names[p] for p in preds]
            st.write(pd.DataFrame({"Prediksi Spesies": pred_species}))
        else:
            st.error("Model belum tersedia.")
