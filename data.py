import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

# ---------------- UI STYLE ----------------

st.set_page_config(
    page_title="Smart Pollution Prediction System",
    layout="wide"
)

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

# df = pd.read_csv("delhi_ncr_aqi_dataset.csv")
df = pd.read_csv("delhi_ncr_aqi_enriched.csv")

df.columns = df.columns.str.strip()

# ---------------- DATE FEATURES ----------------

df['datetime'] = pd.to_datetime(df['datetime'])

df['month'] = df['datetime'].dt.month

df['hour'] = df['datetime'].dt.hour

# ---------------- HANDLE MISSING VALUES ----------------

df = df.fillna(df.mean(numeric_only=True))



# ---------------- FEATURES ----------------

# features = [
#     'pm25',
#     'pm10',
#     'no2',
#     'so2',
#     'co',
#     'o3',
#     'temperature',
#     'humidity',
#     'wind_speed',
#     'visibility',
#     'month',
#     'hour',
#     'latitude',
#     'longitude',
#     'park_count',
#     'hospital_count'
# ]
features = [
    'pm25',
    'pm10',
    'no2',
    'so2',
    'co',
    'o3',
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'latitude',
    'longitude',
    'school_count',
    'hospital_count',
    'park_count',
    'month',
    'hour'
]

X = df[features]

y = df['aqi']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

# =====================================
# MODEL COMPARISON
# =====================================

models = {

    "Linear Regression":
    LinearRegression(),

    "Decision Tree":
    DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest":
    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost":
    XGBRegressor()
}

results = []

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append([

        name,
        mae,
        rmse,
        r2

    ])

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",
        "MAE",
        "RMSE",
        "R² Score"

    ]
)
# =====================================
# FINAL XGBOOST MODEL FOR APP
# =====================================

model = XGBRegressor()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# ---------------- TITLE ----------------

st.title("🌍 Smart Pollution Prediction & Health System")

# ---------------- SIDEBAR ----------------

st.sidebar.header("Enter Pollution Data")

summary = pd.read_csv("station_summary.csv")

selected_station = st.sidebar.selectbox(
    "Select Station",
    summary["station"].unique()
)

station_info = summary[
    summary["station"] == selected_station
].iloc[0]



school_count = int(station_info["school_count"])
hospital_count = int(station_info["hospital_count"])
park_count = int(station_info["park_count"])




st.sidebar.markdown(
    f"""
📍 Station Information

🏫 Schools: {school_count}

🏥 Hospitals: {hospital_count}

🏞️ Parks: {park_count}

⚠️ Risk Level: {station_info['risk_level']}

📊 Avg AQI: {round(station_info['aqi'],2)}
"""
)

pm25 = st.sidebar.slider(
    "PM2.5",
    0,
    500,
    100
)

pm10 = st.sidebar.slider(
    "PM10",
    0,
    500,
    150
)

no2 = st.sidebar.slider(
    "NO2",
    0,
    200,
    50
)

so2 = st.sidebar.slider(
    "SO2",
    0.0,
    500.0,
    50.0
)

co = st.sidebar.slider(
    "CO",
    0.0,
    5.0,
    1.0
)

o3 = st.sidebar.slider(
    "O3",
    0.0,
    500.0,
    50.0
)

temp = st.sidebar.slider(
    "Temperature",
    0,
    50,
    25
)

humidity = st.sidebar.slider(
    "Humidity",
    0,
    100,
    50
)

wind = st.sidebar.slider(
    "Wind Speed",
    0.0,
    10.0,
    2.0
)

visibility = st.sidebar.slider(
    "Visibility",
    0.0,
    20.0,
    10.0
)

month = st.sidebar.selectbox(
    "Month",
    list(range(1,13)),
    index=5
)

hour = st.sidebar.slider(
    "Hour",
    0,
    23,
    12
)

st.sidebar.info(
    "Predict AQI + Health + Pollution Analysis"
)

# ---------------- USER INPUT ----------------


# input_data = pd.DataFrame([{

#     'pm25': pm25,
#     'pm10': pm10,
#     'no2': no2,
#     'so2': so2,
#     'co': co,
#     'o3' : o3,
#     'temperature': temp,
#     'humidity': humidity,
#     'wind_speed': wind,
#     'month': 11,
#     'hour': 10,
#     'latitude': 28.61,
#     'longitude': 77.20,

#     'park_count': park_count,
#     'hospital_count': hospital_count

# }])
input_data = pd.DataFrame([{

    'pm25': pm25,
    'pm10': pm10,
    'no2': no2,
    'so2': so2,
    'co': co,
    'o3': o3,

    'temperature': temp,
    'humidity': humidity,
    'wind_speed': wind,
    'visibility': visibility,

    'latitude': 28.61,
    'longitude': 77.20,

    'school_count': school_count,
    'hospital_count': hospital_count,
    'park_count': park_count,

    'month': month,
    'hour': hour

}])

# ---------------- AQI PREDICTION ----------------

prediction = model.predict(input_data)

aqi = int(prediction[0])

# ---------------- AQI CATEGORY ----------------

def get_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Severe"

category = get_category(aqi)

# ---------------- HEALTH RISKS ----------------

def disease_risk(aqi):

    if aqi <= 50:

        return [
            "No risk",
            "Safe air"
        ]

    elif aqi <= 100:

        return [
            "Mild irritation"
        ]

    elif aqi <= 200:

        return [
            "Asthma risk",
            "Breathing issues"
        ]

    elif aqi <= 300:

        return [
            "Bronchitis",
            "Lung irritation"
        ]

    else:

        return [
            "Severe lung disease",
            "Heart risk"
        ]

# ---------------- ADVICE ----------------

def advice(aqi):

    if aqi <= 50:

        return [
            "Fresh air is safe",
            "Outdoor activity allowed"
        ]

    elif aqi <= 100:

        return [
            "Normal outdoor activity"
        ]

    elif aqi <= 200:

        return [
            "Wear mask",
            "Avoid heavy exercise"
        ]

    else:

        return [
            "Stay indoors",
            "Use air purifier"
        ]

# ---------------- RECOMMENDATIONS ----------------

def recommendations(aqi):

    if aqi <= 50:

        return {
            "health": [
                "Outdoor exercise safe"
            ],

            "diet": [
                "Balanced diet",
                "Stay hydrated"
            ],

            "eco": [
                "Maintain eco habits"
            ]
        }

    elif aqi <= 100:

        return {
            "health": [
                "Avoid long exposure"
            ],

            "diet": [
                "Vitamin C foods",
                "Drink water"
            ],

            "eco": [
                "Use public transport"
            ]
        }

    elif aqi <= 200:

        return {
            "health": [
                "Wear mask"
            ],

            "diet": [
                "Turmeric milk",
                "Green vegetables"
            ],

            "eco": [
                "Avoid burning waste"
            ]
        }

    elif aqi <= 300:

        return {
            "health": [
                "Stay indoors"
            ],

            "diet": [
                "Ginger tea",
                "Tulsi"
            ],

            "eco": [
                "Reduce travel"
            ]
        }

    else:

        return {
            "health": [
                "Strictly indoors"
            ],

            "diet": [
                "Warm fluids",
                "Herbal drinks"
            ],

            "eco": [
                "Avoid vehicles"
            ]
        }

# ---------------- TABS ----------------

# tab1, tab2, tab3, tab4 = st.tabs([
#     "📊 Prediction",
#     "💡 Health",
#     "📈 Analysis",
#     "🗺️ AQI Map"
# ])

# tab1, tab2, tab3, tab4, tab5 = st.tabs([
#     "📊 Prediction",
#     "💡 Health",
#     "📈 Analysis",
#     "🗺️ AQI Map",
#     "🏥 Sensitive Zones"
# ])

tab1, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Prediction & Health",
        "📈 Analysis",
        "🗺️ AQI Map",
        "🏥 Sensitive Zones"
    ]
)

# ---------------- TAB 1 ----------------

with tab1:

    if aqi <= 50:

        color = "#00e676"

    elif aqi <= 100:

        color = "#ffd600"

    elif aqi <= 200:

        color = "#ff9100"

    else:

        color = "#ff1744"

    st.markdown(f"""
    <div class="card">

    <h1 style="color:{color}; text-align:center;">
    AQI: {aqi}
    </h1>

    <h3 style="text-align:center;">
    Category: {category}
    </h3>

    </div>
    """, unsafe_allow_html=True)

    st.progress(min(aqi / 500, 1.0))


    # =====================================
    # HEALTH STATUS
    # =====================================

    st.markdown("---")

    st.subheader("🩺 Health Advisory")

    if aqi <= 50:

       st.success("✅ Healthy Air Quality")

    elif aqi <= 100:

       st.info("🙂 Moderate Air Quality")

    elif aqi <= 200:

       st.warning("😷 Unhealthy For Sensitive Groups")

    elif aqi <= 300:

       st.error("⚠ Very Unhealthy Air")

    else:

       st.error("🚨 Severe Pollution Alert")

    # =====================================
    # HEALTH RISKS
    # =====================================

    st.subheader("⚠ Health Risks")

    risk_col1, risk_col2 = st.columns(2)

    risks = disease_risk(aqi)

    for i, risk in enumerate(risks):

        if i % 2 == 0:

           with risk_col1:
             st.error(f"🚨 {risk}")

        else:

           with risk_col2:
            st.error(f"🚨 {risk}")

    # =====================================
    # ADVICE
    # =====================================

    st.subheader("📢 Immediate Actions")

    for item in advice(aqi):

       st.success(f"✔ {item}")

    # =====================================
    # RECOMMENDATIONS
    # =====================================

    st.subheader("🌿 Recommendations")

    rec = recommendations(aqi)

    col1, col2, col3 = st.columns(3)

    with col1:

       st.info("### 🩺 Health")

       for item in rec["health"]:

         st.write("✔", item)

    with col2:

       st.info("### 🥗 Diet")

       for item in rec["diet"]:

         st.write("✔", item)

    with col3:

       st.info("### 🌱 Sustainability")

       for item in rec["eco"]:

         st.write("✔", item)




with tab4:

    st.subheader("🗺️ Smart Pollution Map")

    with open(
        "delhi_pollution_map.html",
        "r",
        encoding="utf-8"
    ) as f:

        html_data = f.read()

    components.html(
        html_data,
        height=800,
        scrolling=True
    )

with tab3:

    st.header("📈 Machine Learning Model Analysis Dashboard")

    # =====================================
    # KPI CARDS
    # =====================================

    st.subheader("📊 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "MAE",
            round(mae, 2)
        )

    with col2:

        st.metric(
            "RMSE",
            round(rmse, 2)
        )

    with col3:

        # st.metric(
        #     "R² Score",
        #     # round(r2, 2)
        #     round(r2, 4)
        # )
        st.metric(
        "R² Score",
        f"{r2:.6f}"
        )
       


    with col4:

        st.metric(
            "🤖 Model",
            "XGBoost"
        )

    st.markdown("---")

    # =====================================
    # FEATURE IMPORTANCE
    # =====================================

    st.subheader("📈 Feature Importance Analysis")

    importance_df = pd.DataFrame({

        "Feature": features,
        "Importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(

        importance_df,

        x="Importance",
        y="Feature",

        orientation="h",

        color="Importance",

        title="Feature Importance Ranking"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # TOP FEATURES
    # =====================================

    st.subheader("🏆 Top 5 Most Important Features")

    top_features = importance_df.sort_values(
        "Importance",
        ascending=False
    ).head(5)

    st.dataframe(
        top_features,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # ACTUAL VS PREDICTED
    # =====================================

    st.subheader("🎯 Actual vs Predicted AQI")

    prediction_df = pd.DataFrame({

        "Actual AQI": y_test,
        "Predicted AQI": y_pred

    })

    fig = px.scatter(

        prediction_df,

        x="Actual AQI",
        y="Predicted AQI",

        title="Actual vs Predicted AQI",

        opacity=0.6
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    ) 
    st.subheader(
    "🤖 Machine Learning Model Comparison"
    )

    # st.dataframe(
    # results_df,
    # use_container_width=True
    # )
    st.dataframe(
    results_df.style.format({
        "MAE": "{:.6f}",
        "RMSE": "{:.6f}",
        "R² Score": "{:.8f}"
    }),
    use_container_width=True
    )

    st.subheader(
    "🏆 Model Comparison (R² Score)"
    )

    fig = px.bar(

    results_df,

    x="Model",

    y="R² Score",

    color="Model",

    title="Model Performance Comparison"

    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )
    

    st.markdown("---")

   
   



with tab5:

    st.header("🏥 Sensitive Zone Analysis Dashboard")

    summary = pd.read_csv("station_summary.csv")

    # =====================================
    # FILTER
    # =====================================

    risk_filter = st.multiselect(
        "📊 Filter by Risk Level",
        options=summary["risk_level"].unique(),
        default=summary["risk_level"].unique()
    )

    filtered_summary = summary[
        summary["risk_level"].isin(risk_filter)
    ]

    # =====================================
    # KPI CARDS
    # =====================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🏫 Schools",
            int(filtered_summary["school_count"].sum())
        )

    with col2:
        st.metric(
            "🏥 Hospitals",
            int(filtered_summary["hospital_count"].sum())
        )

    with col3:
        st.metric(
            "🏞 Parks",
            int(filtered_summary["park_count"].sum())
        )

    with col4:
        st.metric(
            "🌫 Highest AQI",
            round(filtered_summary["aqi"].max(), 1)
        )

    st.markdown("---")

    # =====================================
    # TOP POLLUTED AREAS
    # =====================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🚨 Top Polluted Areas")

        top_polluted = (
            filtered_summary
            .sort_values("aqi", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_polluted,
            x="aqi",
            y="station",
            orientation="h",
            color="aqi",
            title="Top 10 Polluted Areas",
            hover_data=["risk_level"]
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================
    # SCHOOL ZONES
    # =====================================

    with col2:

        st.subheader("🏫 Top School Zones")

        top_schools = (
            filtered_summary
            .sort_values(
                "school_count",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top_schools,
            x="school_count",
            y="station",
            orientation="h",
            color="school_count",
            hover_data=["aqi", "risk_level"]
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================
    # HOSPITAL ZONES
    # =====================================

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("🏥 Top Hospital Zones")

        top_hospitals = (
            filtered_summary
            .sort_values(
                "hospital_count",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top_hospitals,
            x="hospital_count",
            y="station",
            orientation="h",
            color="hospital_count",
            hover_data=["aqi", "risk_level"]
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================
    # PARK ZONES
    # =====================================

    with col4:

        st.subheader("🏞 Top Park Zones")

        top_parks = (
            filtered_summary
            .sort_values(
                "park_count",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top_parks,
            x="park_count",
            y="station",
            orientation="h",
            color="park_count",
            hover_data=["aqi", "risk_level"]
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # =====================================
    # RISK DISTRIBUTION DONUT CHART
    # =====================================

    st.subheader("📊 Risk Level Distribution")

    risk_counts = (
        filtered_summary["risk_level"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Count"
    ]

    fig = px.pie(
        risk_counts,
        names="Risk Level",
        values="Count",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================
    # DETAIL TABLE
    # =====================================

    st.subheader("📋 Detailed Station Information")

    st.dataframe(
        filtered_summary[
            [
                "station",
                "aqi",
                "school_count",
                "hospital_count",
                "park_count",
                "risk_level"
            ]
        ],
        use_container_width=True
    )