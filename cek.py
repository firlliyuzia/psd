# Core Pkg
import streamlit as st
import os
import pickle  # Import pickle

# EDA Pkgs
import pandas as pd
import numpy as np

# Data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Class labels
class_label = {'low risk': 1, 'mid risk': 2, 'high risk': 3}

@st.cache
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

# Function to get key from value in a dictionary
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

def main():
    st.title("Maternal Health Risk APP")
    st.subheader("Firlli Yuzia Rahmanu | 210411100163")

    st.subheader("Prediction")

    age = st.number_input("Input Age", 0, 100)
    systolic_bp = st.number_input("Input Systolic BP", 0, 500)
    diastolic_bp = st.number_input("Input Diastolic BP", 0, 500)
    bs = st.number_input("Input BS", 0, 100)
    body_temp = st.number_input("Input Body Temp", 0, 100)
    herth_rate = st.number_input("Input Hearth Rate", 0, 500)

    v_age = age
    v_systolic_bp = systolic_bp
    v_diastolic_bp = diastolic_bp
    v_bs = bs
    v_body_temp = body_temp
    v_hearth_rate = herth_rate

    pretty_data = {
        "age": age,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "bs": bs,
        "body_temp": body_temp,
        "hearth": herth_rate,
    }
    
    st.json(pretty_data)

    # Data To Be Used
    sample_data = [v_age, v_systolic_bp, v_diastolic_bp, v_bs, v_body_temp, v_hearth_rate]

    # Normalize the input data using the loaded scaler
    with open('ss.pkl', 'rb') as file:
        normalisasi = pickle.load(file)

    prep_data = normalisasi.transform(np.array(sample_data).reshape(1, -1))

    # Load the Decision Tree model
    with open('decision_tree_model.pkl', 'rb') as dt:
        predictor = pickle.load(dt)

    if st.button('Prediksi'):
        prediction = predictor.predict(prep_data)
        final_result = get_key(prediction, class_label)
        st.success(final_result)

if __name__ == '__main__':
    main()
