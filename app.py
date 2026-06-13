import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Tesla Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# Title
# ==========================================

st.title("🚗 Tesla Stock Price Prediction")

st.markdown("""
This application uses an **LSTM Neural Network Model**
to predict Tesla's next-day stock closing price using
historical stock market data.
""")

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("TSLA.csv")

# ==========================================
# Dashboard Metrics
# ==========================================

latest_price = df["Close"].iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest Tesla Price",
        f"${latest_price:.2f}"
    )

with col2:
    st.metric(
        "Total Records",
        len(df)
    )

with col3:
    st.metric(
        "Highest Price",
        f"${df['Close'].max():.2f}"
    )

# ==========================================
# Dataset Preview
# ==========================================

st.subheader("📊 Dataset Preview")

st.dataframe(df.tail())

# ==========================================
# Interactive Historical Graph
# ==========================================

st.subheader("📈 Historical Closing Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Tesla Close Price",
        line=dict(width=3)
    )
)

fig.update_layout(
    title="Tesla Historical Closing Prices",
    xaxis_title="Trading Days",
    yaxis_title="Price ($)",
    template="plotly_dark",
    hovermode="x unified",
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# Load Trained Model
# ==========================================

model = load_model(
    "Tesla_LSTM_Model.h5",
    compile=False
)

# ==========================================
# Data Preprocessing
# ==========================================

close_prices = df[["Close"]].values

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(
    close_prices
)

# ==========================================
# Prediction Section
# ==========================================

st.subheader("🔮 Next Day Prediction")

if st.button("Predict Next Day Price"):

    last_60_days = scaled_data[-60:]

    X_input = np.array(last_60_days)

    X_input = X_input.reshape(
        1,
        60,
        1
    )

    prediction = model.predict(
        X_input,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction
    )

    predicted_price = prediction[0][0]

    st.metric(
        label="Predicted Tesla Price",
        value=f"${predicted_price:.2f}",
        delta=f"{predicted_price - latest_price:.2f}"
    )

    difference = predicted_price - latest_price

    if difference > 0:
        st.success(
            f"📈 Expected Increase: ${difference:.2f}"
        )
    else:
        st.error(
            f"📉 Expected Decrease: ${abs(difference):.2f}"
        )

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.caption(
    "Built using Streamlit, TensorFlow LSTM, Plotly and Tesla historical stock data."
)