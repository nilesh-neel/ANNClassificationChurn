import streamlit as st
import pandas as pd
import numpy as np  
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder.pkl', 'rb') as f:    
    label_encoder_geo = pickle.load(f)
    
with open('label_encoder.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:     
    scaler = pickle.load(f) 
    
## Streamlit app    

st.title("Customer Churn Prediction")   

# User inputs
geography = st.selectbox("Geography", label_encoder_geo.categories_[0]) 
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age', 18, 100, 1)
balance=st.number_input('Balance', 0.0, 100000.0, 0.0)
credit_score=st.number_input('Credit Score', 0.0, 850.0, 0.0)
estimated_salary=st.number_input('Estimated Salary', 0.0, 1000000.0, 0.0)
tenure=st.slider('Tenure', 0, 10, 1)
num_of_products=st.slider('Number of Products', 1, 4, 1)
has_cr_card=st.selectbox('Has Credit Card', [0, 1])
is_active_member=st.selectbox('Is Active Member', [0, 1])

# Create a DataFrame from user inputs   
input_data = pd.DataFrame({
    # 'Geography': [geography],
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],   
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary] 
})

geo_encoder = label_encoder_geo.transform([[ geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoder, columns=label_encoder_geo.get_feature_names_out(['Geography']))


#Comnbine the encoded geography with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

#Predict the churn probability
prediction = model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")

#Display the prediction
if st.button("Predict"):
    if prediction_proba > 0.5:
        st.success("The customer is likely to churn.")
    else:
        st.error("The customer is unlikely to churn.")
# Display the prediction probability