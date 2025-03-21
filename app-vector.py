import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from opensearchpy import OpenSearch
from io import StringIO
import requests
import joblib
from io import BytesIO
# OpenSearch Configuration
OPENSEARCH_HOST = "https://search-moecarsystem-3t7t2uxwdebide2md7kxull7su.us-east-1.es.amazonaws.com"
INDEX_NAME = "cars_index_new"
USERNAME = "moeuser"
PASSWORD = "Mohamad@123"

@st.cache_resource
def load_scaler(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        scaler = joblib.load(BytesIO(response.content))
        return scaler
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")
        return None

@st.cache_resource
def load_label_encoders(urls: dict):
    """Loads LabelEncoders from S3 URLs and returns a dictionary of encoders."""
    encoders = {}
    for feature, url in urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            encoders[feature] = joblib.load(BytesIO(response.content))
        except Exception as e:
            st.error(f"Failed to load {feature} encoder: {e}")
            encoders[feature] = None
    return encoders


# Connect to OpenSearch
client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_auth=(USERNAME, PASSWORD),
    use_ssl=True,
    verify_certs=True,
    timeout=30
)

# Define categorical and numerical features
CATEGORICAL_FEATURES = ["Category", "Gearbox", "FuelType"]
NUMERICAL_FEATURES = ["FirstReg", "Price", "Mileage", "Performance"]

# URLs for LabelEncoders in S3
label_encoder_urls = {
    "Category": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/Category_encoder.pkl",
    "Gearbox": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/Gearbox_encoder.pkl",
    "FuelTyp": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/FuelTyp_encoder.pkl",
}

label_encoders = load_label_encoders(label_encoder_urls)

# Initialize StandardScaler
scaler_url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/modelvectoring/scaler.pkl"
scaler = load_scaler(scaler_url)
    
# Function to convert user input into vector
def preprocess_input(category, gearbox, fuel_type, first_reg, price, mileage, performance):
    category_encoded = label_encoders["Category"].transform([category])[0]
    gearbox_encoded = label_encoders["Gearbox"].transform([gearbox])[0]
    fuel_type_encoded = label_encoders["FuelTyp"].transform([fuel_type])[0]

    # numerical_values = np.array([[first_reg, price, mileage, performance]])
    # numerical_scaled = scaler.transform(numerical_values)[0] 
    numerical_scaled = scaler.transform([[first_reg, price, mileage, performance]])[0]  # Flatten the result

    return np.concatenate(([category_encoded, gearbox_encoded, fuel_type_encoded], numerical_scaled))

# Function to search similar cars in OpenSearch
def search_similar_cars(query_vector):
    query = {
        "size": 10,
        "query": {
            "knn": {
                "vector": {
                    "vector": query_vector.tolist(),
                    "k": 10
                }
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query)
    return response["hits"]["hits"]

# Function to reverse numerical scaling
def reverse_scaling(scaled_values):
    return scaler.inverse_transform([scaled_values])[0]  # Returns real values


# Streamlit UI
st.title("Car Recommendation System 🚗")
st.write("Find similar cars using OpenSearch 🔍")

# User Inputs
category = st.selectbox("Category", label_encoders["Category"].classes_)
first_reg = st.slider("First Registration Year", 2000, 2025, 2015)
gearbox = st.selectbox("Gearbox", label_encoders["Gearbox"].classes_)
price = st.number_input("Price ($)", min_value=1000, max_value=100000, value=20000)
fuel_type = st.selectbox("Fuel Type", label_encoders["FuelTyp"].classes_)
mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, value=50000)
performance = st.number_input("Performance (HP)", min_value=50, max_value=1000, value=150)

if st.button("Find Similar Cars"):
    query_vector = preprocess_input(category, gearbox, fuel_type, first_reg, price, mileage, performance)
    
    results = search_similar_cars(query_vector)
    
    if results:
        for car in results:
            car_data = car["_source"]
            scaled_values = [car_data["FirstReg"], car_data["Price"], car_data["Mileage"], car_data["Performance"]]
            real_values = reverse_scaling(scaled_values)
            # Assign each value separately
            real_first_reg = real_values[0]
            real_price = real_values[1]
            real_mileage = real_values[2]
            real_performance = real_values[3]
            
            st.write(f"🚗 **{car_data['Make']} {car_data['Model']}** - ${real_price}")
            st.write(f"📏 Mileage: {real_mileage} km | 🔥 Performance: {real_performance} HP")
            st.write("💡 Category:", car_data["Category"])
            st.write("---")
    else:
        st.write("❌ No similar cars found.")
