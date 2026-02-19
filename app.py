import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import timedelta
import copy
from pyswarms.single.global_best import GlobalBestPSO

# =============================
# SET SEED (REPRODUCIBLE)
# =============================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

st.set_page_config(layout="wide")

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.title("Upload Data Saham (Excel)")

uploaded_file = st.sidebar.file_uploader(
    "Upload file Excel (.xlsx) dengan kolom: Date & Close",
    type=["xlsx"]
)

section = st.sidebar.radio(
    "Select Section",
    ["Informasi Data", "In-Depth Analysis", "Hasil Forecast"]
)

# =============================
# LOAD DATA DARI EXCEL
# =============================
@st.cache_data
def load_excel(file):
    df = pd.read_excel(file)

    # Validasi kolom
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("File Excel harus memiliki kolom 'Date' dan 'Close'")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df = df[['Close']]
    df.dropna(inplace=True)

    return df

if uploaded_file is None:
    st.warning("Silakan upload file Excel terlebih dahulu.")
    st.stop()

data = load_excel(uploaded_file)

# =============================
# PREPROCESSING DATA
# =============================
df = data.copy().reset_index()
values = df[['Close']].values

n = len(values)
n_train = int(n * 0.8)

train_values = values[:n_train]
test_values = values[n_train:]

# Normalisasi
scaler = MinMaxScaler()
scaler.fit(train_values)
values_scaled = scaler.transform(values)

# =============================
# SEQUENCE (WINDOW LSTM)
# =============================
WINDOW = 1

def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(values_scaled, WINDOW)

train_end = n_train - WINDOW
X_train = X_all[:train_end]
y_train = y_all[:train_end]
X_test = X_all[train_end:]
y_test = y_all[train_end:]

# =============================
# MODEL LSTM
# =============================
def build_lstm(input_shape, units=16, dropout=0.2, lr=0.001):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(units=units, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

# =============================
# BASELINE LSTM
# =============================
def train_baseline():
    model = build_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=16,
        dropout=0.2,
        lr=0.001
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)

    score = mape(y_true, y_pred)

    return model, history, score, y_pred.flatten(), y_true.flatten()

# =============================
# PSO OPTIMIZATION
# =============================
def train_pso():
    PSO_OPTIONS = {'c1':1.5,'c2':1.5,'w':0.5}
    BOUNDS = ([16,0.0001,8,0.1],[128,0.001,128,0.5])

    split = int(len(X_train)*0.8)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    def objective(particles):
        costs = []
        for p in particles:
            units = int(p[0])
            lr = float(p[1])
            batch = int(p[2])
            dropout = float(p[3])

            try:
                model = build_lstm(
                    input_shape=(X_tr.shape[1], X_tr.shape[2]),
                    units=units,
                    dropout=dropout,
                    lr=lr
                )

                model.fit(X_tr, y_tr, epochs=10, batch_size=batch, verbose=0)
                pred = model.predict(X_val, verbose=0)

                pred_inv = scaler.inverse_transform(pred)
                true_inv = scaler.inverse_transform(y_val)

                mse = mean_squared_error(true_inv, pred_inv)
            except:
                mse = 1e10

            costs.append(mse)

        return np.array(costs)

    optimizer = GlobalBestPSO(
        n_particles=10,
        dimensions=4,
        options=PSO_OPTIONS,
        bounds=BOUNDS
    )

    best_cost, best_pos = optimizer.optimize(objective, iters=10, verbose=False)

    best_units = int(best_pos[0])
    best_lr = float(best_pos[1])
    best_batch = int(best_pos[2])
    best_dropout = float(best_pos[3])

    model = build_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=best_units,
        dropout=best_dropout,
        lr=best_lr
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=best_batch,
        validation_split=0.2,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)

    score = mape(y_true, y_pred)

    return model, history, score, y_pred.flatten(), y_true.flatten()

# =============================
# SESSION STATE
# =============================
if "trained" not in st.session_state:
    st.session_state.trained = False

# =============================
# TRAIN BUTTON
# =============================
if st.sidebar.button("Run Training Model"):
    with st.spinner("Training Model..."):
        st.session_state.model_base, \
        st.session_state.history_base, \
        st.session_state.base_mape, \
        st.session_state.y_pred_base, \
        st.session_state.y_true_base = train_baseline()

        st.session_state.model_pso, \
        st.session_state.history_pso, \
        st.session_state.pso_mape, \
        st.session_state.y_pred_pso, \
        st.session_state.y_true_pso = train_pso()

        st.session_state.trained = True

    st.success("Training selesai!")

# =============================
# SECTION 1
# =============================
if section == "Informasi Data":
    st.subheader("Data Close Price")
    st.line_chart(data['Close'])
    st.write(data.describe())

# =============================
# SECTION 2 ANALYSIS
# =============================
elif section == "In-Depth Analysis":
    if not st.session_state.trained:
        st.warning("Klik Run Training Model dulu")
    else:
        st.subheader("Actual vs Predicted")

        fig, ax = plt.subplots()
        ax.plot(st.session_state.y_true_base, label="Actual")
        ax.plot(st.session_state.y_pred_base, label="Baseline")
        ax.plot(st.session_state.y_pred_pso, label="PSO")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        st.subheader("MAPE Comparison")
        results = pd.DataFrame({
            "Model":["Baseline","PSO"],
            "MAPE":[
                st.session_state.base_mape,
                st.session_state.pso_mape
            ]
        })
        st.dataframe(results)

# =============================
# FORECAST
# =============================
elif section == "Hasil Forecast":
    if not st.session_state.trained:
        st.warning("Klik Run Training Model dulu")
    else:
        st.subheader("Forecast Future")
        future_days = st.slider("Hari Forecast",5,30,7)

        last_window = X_test[-1].copy()
        future_preds = []
        model = st.session_state.model_base

        for _ in range(future_days):
            pred = model.predict(
                last_window.reshape(1,last_window.shape[0],last_window.shape[1]),
                verbose=0
            )
            future_preds.append(pred[0,0])
            last_window = np.roll(last_window,-1)
            last_window[-1] = pred

        future_preds = scaler.inverse_transform(
            np.array(future_preds).reshape(-1,1)
        ).flatten()

        fig, ax = plt.subplots()
        ax.plot(future_preds, label="Forecast")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
