import pickle

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Air Passengers Forecast",
    page_icon="✈️",
    layout="centered",
)

st.markdown(
    """
    <style>
        #MainMenu, header, footer {visibility: hidden;}

        .stApp {
            background: #f5f5f5;
            color: #111111;
        }

        .block-container {
            padding-top: 3rem;
            max-width: 650px;
        }

        div[data-testid="stMarkdownContainer"]:has(.app-title) {
            display: flex;
            justify-content: center;
            width: 100%;
        }

        .app-title {
            font-size: clamp(3.5rem, 4.5vw, 2.5rem);
            font-weight: 800;
            color: #111111;
            margin-bottom: 2rem;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            white-space: nowrap;
        }

        .app-subtitle {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 1.8rem;
            text-align: center;
        }

        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.5rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.07);
        }

        div[data-testid="stNumberInput"] input {
            background: #ffffff !important;
            border: 1.5px solid #d1d5db !important;
            border-radius: 8px !important;
            color: #111111 !important;
            font-size: 0.95rem !important;
            padding: 0.6rem 0.8rem !important;
        }

        div[data-testid="stNumberInput"] input:focus {
            border-color: #1877f2 !important;
            box-shadow: none !important;
        }

        div[data-testid="stNumberInput"] button {
            background: #ffffff !important;
            border: 1.5px solid #d1d5db !important;
            color: #111111 !important;
        }

        div[data-testid="stNumberInput"] button:hover {
            background: #f3f4f6 !important;
            color: #000000 !important;
        }

        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {
            color: #111111 !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1.5px solid #d1d5db !important;
            border-radius: 8px !important;
            color: #111111 !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
            color: #111111 !important;
            background: #ffffff !important;
        }

        div[data-baseweb="popover"] * {
            background-color: #ffffff !important;
            color: #111111 !important;
        }

        div[data-baseweb="popover"] ul li:hover,
        ul[data-baseweb="menu"] li:hover {
            background: #f3f4f6 !important;
        }

        div.stButton {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
            margin-top: 1rem !important;
        }

        div.stButton > button {
            background: #1877f2 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 3rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            white-space: nowrap !important;
            width: auto !important;
        }

        div.stButton > button:hover {
            background: #1461cc !important;
            color: #ffffff !important;
            border: none !important;
        }

        .result-box {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding: 1.2rem;
            border-radius: 10px;
            background-color: #eef6ff;
            border: 1px solid #cfe4ff;
            text-align: center;
        }

        .result-number {
            font-size: 2rem;
            font-weight: 700;
            color: #1877f2;
        }

        .result-label {
            font-size: 0.95rem;
            color: #4b5563;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


saved_model = load_model()
forecast_model = saved_model["model_fit"]
last_date = pd.Timestamp(saved_model["last_date"])

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

st.markdown(
    '<div class="app-title">Air Passengers Forecast</div>',
    unsafe_allow_html=True,
)

with st.container():
    month = st.selectbox("Month", months)

    year = st.number_input(
        "Year",
        min_value=last_date.year + 1,
        max_value=last_date.year + 145,
        value=last_date.year + 1,
        step=1,
    )


    if st.button("Predict"):
        month_num = months.index(month) + 1
        target_date = pd.Timestamp(year=int(year), month=month_num, day=1)

        if target_date <= last_date:
            st.error(
                f"Please select a date after {last_date.strftime('%B %Y')}."
            )
        else:
            steps = (
                (target_date.year - last_date.year) * 12
                + (target_date.month - last_date.month)
            )

            forecast = forecast_model.get_forecast(steps=steps)
            prediction = forecast.predicted_mean.iloc[-1]
            interval = forecast.conf_int().iloc[-1]

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">
                        Forecasted Passengers {month} {int(year)}
                    </div>
                    <div class="result-number">
                        {prediction:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )