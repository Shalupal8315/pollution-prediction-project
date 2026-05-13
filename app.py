import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1e1e2f, #2c2c54);
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("delhi_ncr_aqi_dataset.csv")
df.columns = df.columns.str.strip()

df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour

df = df.fillna(df.mean(numeric_only=True))

# ---------------- FEATURES ----------------
features = [
    'pm25','pm10','no2','co',
    'temperature','humidity','wind_speed',
    'month','hour','latitude','longitude'
]

X = df[features]
y = df['aqi']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = XGBRegressor()
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ---------------- TITLE ----------------
st.title("🌍 Smart Pollution Prediction & Health System")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Enter Pollution Data")

pm25 = st.sidebar.slider("PM2.5", 0, 500, 100)
pm10 = st.sidebar.slider("PM10", 0, 500, 150)
no2 = st.sidebar.slider("NO2", 0, 200, 50)
co = st.sidebar.slider("CO", 0.0, 5.0, 1.0)
temp = st.sidebar.slider("Temperature", 0, 50, 25)
humidity = st.sidebar.slider("Humidity", 0, 100, 50)
wind = st.sidebar.slider("Wind Speed", 0.0, 10.0, 2.0)

st.sidebar.info("Predict AQI + Get Health, Diet & Sustainability Advice")

# ---------------- USER INPUT ----------------
input_data = pd.DataFrame([{
    'pm25': pm25,
    'pm10': pm10,
    'no2': no2,
    'co': co,
    'temperature': temp,
    'humidity': humidity,
    'wind_speed': wind,
    'month': 11,
    'hour': 10,
    'latitude': 28.61,
    'longitude': 77.20
}])

prediction = model.predict(input_data)
aqi = int(prediction[0])

# ---------------- CATEGORY ----------------
def get_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Severe"

category = get_category(aqi)

# ---------------- FUNCTIONS ----------------
def disease_risk(aqi):
    if aqi <= 50:
        return ["No risk", "Safe air"]
    elif aqi <= 100:
        return ["Mild irritation"]
    elif aqi <= 200:
        return ["Asthma risk", "Breathing issues"]
    elif aqi <= 300:
        return ["Bronchitis", "Lung irritation"]
    else:
        return ["Severe lung disease", "Heart risk"]

def advice(aqi):
    if aqi <= 50:
        return ["Go outside", "Fresh air is safe"]
    elif aqi <= 100:
        return ["Normal outdoor activity"]
    elif aqi <= 200:
        return ["Wear mask", "Avoid heavy exercise"]
    else:
        return ["Stay indoors", "Use air purifier"]

def recommendations(aqi):
    if aqi <= 50:
        return {
            "health": ["Clean air", "Outdoor exercise safe"],
            "diet": ["Balanced diet", "Stay hydrated"],
            "eco": ["Maintain eco habits"]
        }
    elif aqi <= 100:
        return {
            "health": ["Avoid long exposure"],
            "diet": ["Vitamin C foods", "Drink water"],
            "eco": ["Use public transport"]
        }
    elif aqi <= 200:
        return {
            "health": ["Wear mask"],
            "diet": ["Turmeric milk", "Green vegetables"],
            "eco": ["Avoid burning waste"]
        }
    elif aqi <= 300:
        return {
            "health": ["Stay indoors"],
            "diet": ["Ginger tea", "Tulsi"],
            "eco": ["Reduce travel"]
        }
    else:
        return {
            "health": ["Strictly indoors"],
            "diet": ["Warm fluids", "Herbal drinks"],
            "eco": ["Avoid vehicles"]
        }

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "💡 Health", "📈 Analysis"])

# ---------------- TAB 1 ----------------
with tab1:

    if aqi <= 50: color = "#00e676"
    elif aqi <= 100: color = "#ffd600"
    elif aqi <= 200: color = "#ff9100"
    else: color = "#ff1744"

    st.markdown(f"""
    <div class="card">
    <h1 style="color:{color}; text-align:center;">AQI: {aqi}</h1>
    <h3 style="text-align:center;">Category: {category}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(aqi/500,1.0))

# ---------------- TAB 2 ----------------
with tab2:

    st.subheader("⚠️ Health Risks")
    st.table(pd.DataFrame({"Effects": disease_risk(aqi)}))

    st.subheader("📢 Advice")
    for a in advice(aqi):
        st.write("✔️", a)

    st.subheader("🌿 Recommendations")
    rec = recommendations(aqi)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🩺 Health")
        for i in rec["health"]:
            st.write("✔️", i)

    with col2:
        st.markdown("### 🥗 Diet")
        for i in rec["diet"]:
            st.write("✔️", i)

    with col3:
        st.markdown("### 🌱 Sustainability")
        for i in rec["eco"]:
            st.write("✔️", i)

# ---------------- TAB 3 ----------------
with tab3:

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", round(mae, 2))

    with col2:
        st.metric("RMSE", round(rmse, 2))

    with col3:
        st.metric("R² Score", round(r2, 2))

    st.subheader("📈 Feature Importance")

    fig, ax = plt.subplots()
    ax.barh(features, model.feature_importances_)
    ax.set_title("Impact on AQI")

    st.pyplot(fig)

    st.subheader("🔍 Interpretation")

    if r2 > 0.8:
        st.success("Model is performing well")
    elif r2 > 0.5:
        st.warning("Model is average")
    else:
        st.error("Model needs improvement")