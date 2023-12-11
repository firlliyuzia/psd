
    # Core Pkg
import streamlit as st
import os
import joblib


# EDA Pkgs
import pandas as pd
import numpy as np

# Data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
# import seaborn as sns

@st.cache
def load_data(dataset):
  df = pd.read_csv(dataset)
  return df

def load_predict_model(model_file) :
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


class_label = {'low risk': 1, 'mid risk': 2, 'high risk': 3}
    # Get the Keys

def get_value(val,my_dict):
  for key ,value in my_dict.items():
    if val == key:
      return value

# Find the Key From Dictionary
def get_key(val,my_dict):
  for key ,value in my_dict.items():
    if val == value:
      return key


def main():
  st.title("Maternal Health Risk APP")
  st.subheader("Kelompok 7")
  st.write("Siti Ma'rifatul Khoiriyah | 210411100007")
  st.write("Firlli Yuzia Rahmanu | 210411100163")
  st.write("Robiatul Adawiyah | 210411100187")

  #menu
  menu = ["EDA", "Prediction"]
  choice = st.sidebar.selectbox("Pilih Menu", menu)

  if choice == 'EDA':
    st.subheader("EDA")

    dataku = load_data('data/maternal_healt_risk.csv') 
    st.dataframe(dataku.head(15))

    if st.checkbox("Show Shape") :
      st.write(dataku.shape)

    if st.checkbox("Pie Chart"):
      fig, ax = plt.subplots()
      dataku['RiskLevel'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
      st.pyplot(fig)          

    if st.checkbox("Value Count Plot"):
      fig, ax = plt.subplots()
      dataku['RiskLevel'].value_counts().plot(kind='bar', ax=ax)
      st.pyplot(fig)




  if choice == 'Prediction':
    st.subheader("Prediction")

    age = st.number_input("Input Age", 0,100)
    systolic_bp = st.number_input("Input Systolic BP", 0,500)
    diastolic_bp = st.number_input("Input Diastolic BP", 0,500)
    bs = st.number_input("Input BS", 0,100)
    body_temp = st.number_input("Input Body Temp", 0,100)
    herth_rate = st.number_input("Input Hearth Rate", 0,500)

    v_age = age
    v_systolic_bp = systolic_bp
    v_diastolic_bp = diastolic_bp
    v_bs = bs
    v_body_temp = body_temp
    v_hearth_rate = herth_rate


    pretty_data = {
    "age":age,
    "systolic_bp":systolic_bp,
    "":diastolic_bp,
    "bs":bs,
    "body_temp":body_temp,
    "hearth":herth_rate,
    }
    st.subheader("Options Selected")
    st.json(pretty_data)

    st.subheader("Input Data Result")
    # Data To Be Used
    sample_data = [v_age,v_systolic_bp,v_diastolic_bp,v_bs,v_body_temp,v_hearth_rate]
    st.write(sample_data)
    button = st.button('Prediksi')
    if button:
        with open('pickle/zscore2_normalisasi.pkl', 'rb') as file:
            normalisasi = pickle.load(file)
        prep_data = normalisasi.transform(np.array(sample_data).reshape(1, -1))
        st.write('Normalisasi : ')
        st.write(prep_data)
        with open('pickle/random_forest.pkl', 'rb') as dt:
            predictor = pickle.load(dt)
            # Hanya menggunakan model Random Forest
            prediction = predictor.predict(prep_data)


            # Jika prediction adalah bilangan bulat yang merujuk pada kelas
            final_result = get_key(prediction, class_label)
            st.success(final_result)





if __name__ == '__main__':
  main()
