import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# =============================
# SIDEBAR INPUT
# =============================
today = datetime.date.today()

start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.date(2019, 1, 1),   # default saja, tetap bisa diubah
    min_value=datetime.date(2000, 1, 1),
    max_value=today
)

end_date = st.sidebar.date_input(
    "End Date",
    value=today,
    min_value=datetime.date(2000, 1, 1),
    max_value=today
)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data(ticker, start, end):

    df = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    # pastikan hanya Close
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close'].to_frame()

    df = df[['Close']].dropna()

    return df

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error(f"Data {ticker} tidak ditemukan. Coba ticker lain.")
    st.stop()

# =============================
# SECTION 1 : INFORMASI DATA
# =============================
if section == "Informasi Data":

    st.subheader(f"Pergerakan Harga Saham {ticker}")
    st.line_chart(data)

    st.subheader("Statistik Deskriptif Harga Close")
    desc = data['Close'].describe()
    st.table(desc)

# =============================
# SECTION 2 : IN DEPTH ANALYSIS
# =============================
elif section == "In-Depth Analysis":

    st.subheader("Perbandingan Model")

    # placeholder hasil training
    mape_baseline = 5.2
    mape_ga = 4.6
    mape_pso = 4.3

    results = pd.DataFrame({
        "Model":["Baseline","GA","PSO"],
        "MAPE":[mape_baseline, mape_ga, mape_pso]
    })

    st.table(results)

    # validation loss dummy
    fig, ax = plt.subplots()
    ax.plot(np.random.rand(50), label="Baseline")
    ax.plot(np.random.rand(50), label="GA")
    ax.plot(np.random.rand(50), label="PSO")
    ax.set_title("Validation Loss")
    ax.legend()
    st.pyplot(fig)

# =============================
# SECTION 3 : FORECAST
# =============================
elif section == "Hasil Forecast":

    st.subheader("Forecast")

    horizon = st.slider("Forecast berapa hari ke depan", 1, 90, 30)

    # dummy forecast
    last_price = data['Close'].iloc[-1]
    forecast = last_price + np.cumsum(np.random.randn(horizon))

    forecast_df = pd.DataFrame({
        "Forecast": forecast
    })

    st.line_chart(forecast_df)



