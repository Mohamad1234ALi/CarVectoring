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
OPENSEARCH_HOST = "https://search-moesystemcar-dnrec2g6qpqy5s5qtlujs7mqly.us-east-1.es.amazonaws.com"
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

# Load label encoders (these must match what was used in your index)
label_encoders = {
    "Category": LabelEncoder().fit(["OffRoad", "Van", "Limousine", "Estate Car", "Small car", "Sport Car"]),
    "Gearbox": LabelEncoder().fit(["Manual", "Automatic"]),
    "FuelType": LabelEncoder().fit(["Petrol", "Diesel", "Electric", "Hybrid"]),
}


# Load MinMaxScaler (these must match your stored embeddings)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array([[2000, 1000, 0, 50], [2025, 100000, 300000, 1000]]))  # Adjust based on your dataset

# Function to convert user input into vector
def preprocess_input(category, gearbox, fuel_type, first_reg, price, mileage, performance):
    category_encoded = label_encoders["Category"].transform([category])[0]
    gearbox_encoded = label_encoders["Gearbox"].transform([gearbox])[0]
    fuel_type_encoded = label_encoders["FuelType"].transform([fuel_type])[0]

    numerical_values = np.array([[first_reg, price, mileage, performance]])
    numerical_scaled = scaler.transform(numerical_values)[0] 

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

# Streamlit UI
st.title("Car Recommendation System 🚗")
st.write("Find similar cars using OpenSearch 🔍")

# User Inputs
category = st.selectbox("Category", label_encoders["Category"].classes_)
first_reg = st.slider("First Registration Year", 2000, 2025, 2015)
gearbox = st.selectbox("Gearbox", label_encoders["Gearbox"].classes_)
price = st.number_input("Price ($)", min_value=1000, max_value=100000, value=20000)
fuel_type = st.selectbox("Fuel Type", label_encoders["FuelType"].classes_)
mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, value=50000)
performance = st.number_input("Performance (HP)", min_value=50, max_value=1000, value=150)

if st.button("Find Similar Cars"):
    query_vector = preprocess_input(category, gearbox, fuel_type, first_reg, price, mileage, performance)
    
    results = search_similar_cars(query_vector)

    if results:
        for car in results:
            car_data = car["_source"]
            
            real_category = car_data["Category"]
            real_gearbox = car_data["Gearbox"]
            real_fuel_type = car_data["FuelType"]
            real_first_reg = car_data["FirstReg"]
            real_price = car_data["Price"]
            real_mileage = car_data["Mileage"]
            real_performance = car_data["Performance"]
        
            st.write(f"🚗 **{car_data['Make']} {car_data['Model']}** - ${real_price}")
            st.write(f"📏 Mileage: {real_mileage} km | 🔥 Performance: {real_performance} HP")
            st.write(f"💡 Category: {real_category} | Gearbox: {real_gearbox} | Fuel Type: {real_fuel_type}")
            st.write(f"📅 First Registration: {real_first_reg}")
            st.write("---")

    
    else:
        st.write("❌ No similar cars found.")
