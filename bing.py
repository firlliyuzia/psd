import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pickled model and normalizer
model = pickle.load(open('best_model.pkl', 'rb'))
normalizer = pickle.load(open('normalisasi_ini.pkl', 'rb'))
class_label = {0: 'low risk', 1: 'mid risk', 2: 'high risk'}  # Memperbarui pemetaan class_label

# Create a function to make predictions
def predict(data):
    # Normalize the data
    data = normalizer.transform(data)
    # Make a prediction
    prediction = model.predict(data)
    return prediction[0]  # Mengambil prediksi pertama dari array

# Create the Streamlit app
def main():
    # Add a title to the app
    st.title('Prediction App')
    # Add a description to the app
    st.write('maternal health risk')
    # Add input fields to the app
    age = st.number_input("Input Age", 0, 100)
    systolic_bp = st.number_input("Input Systolic BP", 0, 500)
    diastolic_bp = st.number_input("Input Diastolic BP", 0, 500)
    bs = st.number_input("Input BS", step=0.01)
    body_temp = st.number_input("Input Body Temp", 0, 200)
    HeartRate = st.number_input("Input Heart Rate", 0, 500)  # "Hearth Rate" diganti menjadi "Heart Rate"
    # Create a list of input data
    data = [[age, systolic_bp, diastolic_bp, bs, body_temp, HeartRate]]
    # Make a prediction and display it
    if st.button('Predict'):
        prediction = predict(data)
        class_name = class_label[prediction]
        st.write(f'The predicted class is {class_name}.')

if __name__ == '__main__':
    main()
