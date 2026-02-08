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
import datetime

st.sidebar.title("Stock Settings")

ticker_input = st.sidebar.text_input(
    "Masukkan ticker saham (contoh: BBCA)",
    "SIDO"
)

# otomatis tambah .JK jika belum ada
if ".JK" not in ticker_input:
    ticker = ticker_input.upper() + ".JK"
else:
    ticker = ticker_input.upper()

today = datetime.date.today()

start_date = st.sidebar.date_input(
    "Start Date",
    datetime.date(2019,1,1)
)

end_date = st.sidebar.date_input(
    "End Date",
    today
)

section = st.sidebar.radio(
    "Select Section",
    ["Informasi Data", "In-Depth Analysis", "Hasil Forecast"]
)


# =============================
# LOAD DATA
# =============================
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return pd.DataFrame()

    # Jika MultiIndex kolom
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ambil hanya Close
    df = df[['Close']].copy()

    df.dropna(inplace=True)

    return df

# =============================
# SECTION 1 : INFORMASI DATA
# =============================
if section == "Informasi Data":

st.subheader(f"Pergerakan Harga Saham {ticker}")
st.line_chart(data['Close'])

st.subheader("Statistik Deskriptif (Close)")
st.write(data['Close'].describe())

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





