import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk

## loading the Random Forest Classifier Model
with open("random_forest_model.pkl",'rb') as model_file:
    rfc = pk.load(model_file)

st.set_page_config(
    page_title="Warranty Claimns Fraud Prediction",
    page_icon="🤷🏻",
    layout="wide"
)

## lable Encoding
region_labels = {'South': 4, 
                 'North': 1, 
                 'North East': 2, 
                 'North West': 3, 
                 'South East': 5, 
                 'South West': 6, 
                 'West': 7, 
                 'East': 0}

state_labels = {'Karnataka': 10, 
                'Haryana': 6, 
                'Tamil Nadu': 16, 
                'Jharkhand': 9, 
                'Kerala': 11, 
                'Andhra Pradesh': 0, 
                'Bihar': 2, 
                'Gujarat': 5, 
                'Delhi': 3, 
                'Maharashtra': 13, 
                'West Bengal': 19, 
                'Goa': 4, 
                'Jammu and Kashmir': 8, 
                'Assam': 1, 
                'Rajasthan': 15, 
                'Madhya Pradesh': 12, 
                'Uttar Pradesh': 18, 
                'Tripura': 17, 
                'Himachal Pradesh': 7, 
                'Orissa': 14}

area_labels = {'Urban': 1, 
               'Rural': 0}

city_labels = {'Bangalore': 2, 
               'Chandigarh': 5, 
               'Chennai': 6, 
               'Ranchi': 21, 
               'Kochi': 11, 
               'Hyderabad': 9, 
               'Patna': 18, 
               'Purnea': 20, 
               'Vadodara': 24, 
               'New Delhi': 16, 
               'Mumbai': 15, 
               'Ahmedabad': 1, 
               'Pune': 19, 
               'Kolkata': 12, 
               'Vizag': 26, 
               'Panaji': 17, 
               'Srinagar': 23, 
               'Guwhati': 8, 
               'Jaipur': 10, 
               'Bhopal': 3, 
               'Meerut': 14, 
               'Delhi': 7, 
               'Agartala': 0, 
               'Shimla': 22, 
               'Bhubaneswar': 4, 
               'Vijayawada': 25, 
               'Lucknow': 13}

consumer_profile_labels = {'Business': 0, 
                           'Personal': 1}

product_category_labels = {'Entertainment': 0, 
                           'Household': 1}

product_type_labels = {'TV': 1, 
                       'AC': 0}

ac_1001_issue_labels = {'No Issue': 0, 
                        'Repair': 1, 
                        'Replacement': 2}

ac_1002_issue_labels = {'No Issue': 0, 
                        'Repair': 1, 
                        'Replacement': 2}

ac_1003_issue_labels = {'No Issue': 0, 
                        'Repair': 1, 
                        'Replacement': 2}

tv_2001_issue_labels = {'No Issue': 0, 
                        'Repair': 1, 
                        'Replacement': 2}

tv_2002_issue_labels = {'No Issue': 0, 
                        'Repair': 1, 
                        'Replacement': 2}

tv_2003_issue_labels = {'No Issue': 0, 
                        'Repair': 1, 
                        'Replacement': 2}

purchased_from_labels = {'Manufacturer': 2, 
                         'Dealer': 0, 
                         'Internet': 1}

purpose_labels = {'Complaint': 1, 
                  'Claim': 0, 
                  'Other': 2}

# User Interface
st.title("Warranty Claimns Fraud Prediction 🤷🏻🙅")
st.sidebar.title("Warranty Claimns Fraud Prediction")
st.sidebar.header("Fill the details...")
user_input = {}

### input features

## Categorical Features

# User input for Region
user_input_Region = st.sidebar.selectbox("Region", options=['South', 'North', 'North East', 'North West', 'South East', 'South West', 'West', 'East'])

# User input for State
user_input_State = st.sidebar.selectbox("State", options=['Karnataka', 'Haryana', 'Tamil Nadu', 'Jharkhand', 'Kerala', 'Andhra Pradesh', 'Bihar', 'Gujarat', 'Delhi', 'Maharashtra', 'West Bengal', 'Goa', 'Jammu and Kashmir', 'Assam', 'Rajasthan', 'Madhya Pradesh', 'Uttar Pradesh', 'Tripura', 'Himachal Pradesh', 'Orissa'])

# User input for Area
user_input_Area = st.sidebar.radio("Area", options=['Urban', 'Rural'])

# User input for City
user_input_City = st.sidebar.selectbox("City", options=['Bangalore', 'Chandigarh', 'Chennai', 'Ranchi', 'Kochi', 'Hyderabad', 'Patna', 'Purnea', 'Vadodara', 'New Delhi', 'Mumbai', 'Ahmedabad', 'Pune', 'Kolkata', 'Vizag', 'Panaji', 'Srinagar', 'Guwhati', 'Jaipur', 'Bhopal', 'Meerut', 'Delhi', 'Agartala', 'Shimla', 'Bhubaneswar', 'Vijayawada', 'Lucknow'])

# User input for Consumer Profile
user_input_Consumer_profile = st.sidebar.radio("Consumer Profile", options=['Business', 'Personal'])

# User input for Product Category
user_input_Product_category = st.sidebar.radio("Product Category", options=['Entertainment', 'Household'])

# User input for Product Type
user_input_Product_type = st.sidebar.radio("Product Type", options=['TV', 'AC'])

# User input for AC Issues
user_input_AC_1001_Issue = st.sidebar.radio("AC 1001 Issue", options=['No Issue', 'Repair', 'Replacement'])
user_input_AC_1002_Issue = st.sidebar.radio("AC 1002 Issue", options=['No Issue', 'Repair', 'Replacement'])
user_input_AC_1003_Issue = st.sidebar.radio("AC 1003 Issue", options=['No Issue', 'Repair', 'Replacement'])

# User input for TV Issues
user_input_TV_2001_Issue = st.sidebar.radio("TV 2001 Issue", options=['No Issue', 'Repair', 'Replacement'])
user_input_TV_2002_Issue = st.sidebar.radio("TV 2002 Issue", options=['No Issue', 'Repair', 'Replacement'])
user_input_TV_2003_Issue = st.sidebar.radio("TV 2003 Issue", options=['No Issue', 'Repair', 'Replacement'])

# User input for Purchased From
user_input_Purchased_from = st.sidebar.radio("Purchased From", options=['Manufacturer', 'Dealer', 'Internet'])

# User input for Purpose
user_input_Purpose = st.sidebar.radio("Purpose", options=['Complaint', 'Claim', 'Other'])

## Numrical Features

# User input for Claim_Value
user_input_Claim_Value = st.sidebar.number_input("Claim Value", value=20000.0)

# User input for Service Centre
user_input_Service_Centre = st.sidebar.number_input("Service Centre", value=12)

# User input for Product Age
user_input_Product_Age = st.sidebar.number_input("Product Age", value=10)

# User input for Call Details
user_input_Call_details = st.sidebar.number_input("Call Details", value=1.0)

## MAKING PAREDICTION ON INPUT DATA

def make_prediction():
    # converting the user input to a numpy array.
    input_array = np.array([[region_labels[user_input_Region],
                            state_labels[user_input_State],
                            area_labels[user_input_Area],
                            city_labels[user_input_City],
                            consumer_profile_labels[user_input_Consumer_profile],
                            product_category_labels[user_input_Product_category],
                            product_type_labels[user_input_Product_type],
                            ac_1001_issue_labels[user_input_AC_1001_Issue],
                            ac_1002_issue_labels[user_input_AC_1002_Issue],
                            ac_1003_issue_labels[user_input_AC_1003_Issue],
                            tv_2001_issue_labels[user_input_TV_2001_Issue],
                            tv_2002_issue_labels[user_input_TV_2002_Issue],
                            tv_2003_issue_labels[user_input_TV_2003_Issue],
                            purchased_from_labels[user_input_Purchased_from],
                            purpose_labels[user_input_Purpose],
                            user_input_Claim_Value,
                            user_input_Service_Centre,
                            user_input_Product_Age,
                            user_input_Call_details]])
    
    # model prediction
    prediction = rfc.predict(input_array)
    st.write("Model Prediction 🖥️: ",prediction[0])

    # checking the conditions using if and else.
    if prediction[0] == 0:
        st.success("Genuine!!🆗✅👌🏻")
    else:
        st.error("Fraudulent!!🚫❌🙆🏻")

# if user clicks on submit button then it will predict the output.
if st.sidebar.button("Submit ➡️"):
    with st.spinner("Predicting 🫸🏻....."):
        make_prediction()