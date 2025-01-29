# importing sources
import streamlit as st
import pickle
import numpy as np
import requests
import pandas as pd
import streamlit as st
st.write(st.secrets)

# OpenWeather function
def get_weather_data(location):
    api_key = st.secrets["api_keys"]["openweather"]  # Secure API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        return data['main']['temp']  
    else:
        return None

# Load model and label encoders
try:
    with open("model_file.pkl", "rb") as f:
        model, label_encoders = pickle.load(f)
except Exception:
    st.error("Failed to load the model file. Please check the file.")
    st.stop()

# Check label encoders
if "Crop Stage" not in label_encoders or "Any Cat Event" not in label_encoders:
    st.error("Label encoders are not properly loaded. Please check the model file.")
    st.stop()

# Streamlit Title
st.title("Farmer Advisory System")

# User inputs
location = st.text_input("Enter Location (e.g., Delhi, Gujarat)")
crop_stage = st.selectbox("Select Crop Stage", label_encoders['Crop Stage'].classes_)
cat_event = st.selectbox("Select Category Event", label_encoders['Any Cat Event'].classes_)

# Temperature Fetch
temperature = None
if location:
    temperature = get_weather_data(location)
    if temperature is not None:
        st.write(f"Current Temperature in {location}: {temperature}°C")
    else:
        st.write("Could not fetch weather data. Please try again.")

# Advisory Button
if st.button("Get<>Advisory"):
    try:
        if temperature is not None:
            crop_stage_encoded = label_encoders['Crop Stage'].transform([crop_stage])[0]
            cat_event_encoded = label_encoders['Any Cat Event'].transform([cat_event])[0]

            input_features = np.array([[crop_stage_encoded, cat_event_encoded]])
            prediction = model.predict(input_features)
            advisory = label_encoders['Agro Advisory'].inverse_transform(prediction)[0]

            st.write(f"Advisory: {advisory}")

            advisory_df = pd.DataFrame({
                "Location": [location],
                "Crop Stage": [crop_stage],
                "Category Event": [cat_event],
                "Temperature (°C)": [temperature],
                "Advisory": [advisory]
            })

            csv = advisory_df.to_csv(index=False)
            json = advisory_df.to_json(orient="records", lines=True)

            st.download_button(
                label="Download Advisory as CSV",
                data=csv,
                file_name="farmer_advisory.csv",
                mime="text/csv"
            )
            st.download_button(
                label="Download Advisory as JSON",
                data=json,
                file_name="farmer_advisory.json",
                mime="application/json"
            )
        else:
            st.write("Unable to fetch temperature. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
