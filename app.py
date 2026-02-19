import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import timedelta
import copy
from pyswarms.single.global_best import GlobalBestPSO

# =========================
# CONFIG & SEED
# =========================
st.set_page_config(layout="wide")
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Input Data Saham (Excel)")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel (Kolom: Date & Close)",
    type=["xlsx"]
)

section = st.sidebar.radio(
    "Menu",
    ["Proses Data", "Training & Evaluasi", "Forecast"]
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_excel(file):
    df = pd.read_excel(file)

    # Standarisasi nama kolom
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns or "Close" not in df.columns:
        st.error("File harus memiliki kolom: Date dan Close")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[["Date", "Close"]]
    df.dropna(inplace=True)

    return df

if uploaded_file is None:
    st.warning("Silakan upload file Excel terlebih dahulu.")
    st.stop()

df_raw = load_excel(uploaded_file)

# =========================
# PREPROCESSING (DITAMPILKAN)
# =========================
df = df_raw.copy()

# Pastikan kolom benar
df.columns = [c.strip() for c in df.columns]

if "Date" not in df.columns or "Close" not in df.columns:
    st.error("File harus memiliki kolom: Date dan Close")
    st.stop()

df = df.reset_index(drop=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df = df[['Close']]
df = df.reset_index()
df.columns = ['Date', 'Close']
df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')

feature_cols = ["Close"]
target_col = "Close"
window = 1  # SAMA PERSIS KODE LAMAMU

data_features = df[feature_cols].values
data_target = df[[target_col]].values

values = df[['Close']].values
n = len(values)
n_train = int(n * 0.8)

train_values = values[:n_train]
test_values = values[n_train:]

# Scaling
scaler_X = MinMaxScaler().fit(data_features[:n_train])
scaler_y = MinMaxScaler().fit(data_target[:n_train])

Xs = scaler_X.transform(data_features)
ys = scaler_y.transform(data_target)

# Sequence LSTM
def make_sequences(X_scaled, y_scaled, window):
    X_seq, y_seq = [], []
    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i-window:i])
        y_seq.append(y_scaled[i])
    return np.array(X_seq), np.array(y_seq)

X_seq_all, y_seq_all = make_sequences(Xs, ys, window=window)

train_end_idx = n_train - window
X_train = X_seq_all[:train_end_idx]
y_train = y_seq_all[:train_end_idx]
X_test = X_seq_all[train_end_idx:]
y_test = y_seq_all[train_end_idx:]

# =============================
# HYPERPARAMETER BASELINE (SAMA)
# =============================
BASE_UNITS = 16
BASE_DROPOUT = 0.5
BASE_BATCH = 64
BASE_EPOCHS = 100
BASE_LR = 0.001

# =========================
# MODEL
# =========================
def build_lstm_model(input_shape, units=16, dropout=0.01, lr=1e-3):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(units=units, input_shape=input_shape))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

# =========================
# TRAIN BASELINE
# =========================
@st.cache_resource
def train_baseline():
    model_base = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=BASE_UNITS,
        dropout=BASE_DROPOUT,
        lr=BASE_LR
    )

    history_base = model_base.fit(
        X_train, y_train,
        epochs=BASE_EPOCHS,
        batch_size=BASE_BATCH,
        validation_split=0.2,
        verbose=0
    )

    y_pred_scaled = model_base.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()

    base_mape = mape(y_true, y_pred)

    return model_base, history_base, base_mape, y_pred, y_true

def mape(y_true, y_pred):
# =========================
# FITNESS FUNCTION (UNTUK PSO & GA)
# =========================
def fitness_lstm(params):
    """
    params = [units, dropout, lr, batch_size]
    """
    units = int(params[0])
    dropout = float(params[1])
    lr = float(params[2])
    batch_size = int(params[3])

    # Batas keamanan (supaya tidak error)
    units = max(4, min(128, units))
    dropout = max(0.0, min(0.7, dropout))
    lr = max(1e-5, min(0.01, lr))
    batch_size = max(8, min(128, batch_size))

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=units,
        dropout=dropout,
        lr=lr
    )

    model.fit(
        X_train, y_train,
        epochs=30,          # lebih kecil agar PSO/GA tidak lama
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()

    return mape(y_true, y_pred)

# =========================
# PSO OPTIMIZATION
# =========================
@st.cache_resource
def train_pso():
    # Batas hyperparameter: [units, dropout, lr, batch_size]
    lb = [8, 0.0, 1e-4, 16]
    ub = [64, 0.6, 0.01, 128]

    def objective_function(x):
        scores = []
        for particle in x:
            score = fitness_lstm(particle)
            scores.append(score)
        return np.array(scores)

    optimizer = GlobalBestPSO(
        n_particles=5,      # kecil biar tidak lama
        dimensions=4,
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.5},
        bounds=(np.array(lb), np.array(ub))
    )

    best_cost, best_pos = optimizer.optimize(
        objective_function,
        iters=3   # kecil agar tidak lama di Streamlit
    )

    best_units = int(best_pos[0])
    best_dropout = float(best_pos[1])
    best_lr = float(best_pos[2])
    best_batch = int(best_pos[3])

    # Train model terbaik PSO
    best_model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=best_units,
        dropout=best_dropout,
        lr=best_lr
    )

    history = best_model.fit(
        X_train, y_train,
        epochs=BASE_EPOCHS,
        batch_size=best_batch,
        validation_split=0.2,
        verbose=0
    )

    y_pred_scaled = best_model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()

    best_mape = mape(y_true, y_pred)

    best_params = {
        "units": best_units,
        "dropout": best_dropout,
        "lr": best_lr,
        "batch_size": best_batch
    }

    return best_model, history, best_mape, y_pred, y_true, best_params

# =========================
# GENETIC ALGORITHM (GA)
# =========================
@st.cache_resource
def train_ga():
    pop_size = 6
    generations = 3
    mutation_rate = 0.2

    def create_individual():
        return [
            random.randint(8, 64),          # units
            random.uniform(0.0, 0.6),       # dropout
            random.uniform(1e-4, 0.01),     # lr
            random.randint(16, 128)         # batch
        ]

    population = [create_individual() for _ in range(pop_size)]

    for gen in range(generations):
        fitness_scores = [fitness_lstm(ind) for ind in population]

        # Seleksi (elitism)
        sorted_pop = [
            x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])
        ]
        population = sorted_pop[:3]

        # Crossover
        children = []
        while len(children) < 3:
            p1, p2 = random.sample(population, 2)
            cp = random.randint(1, 3)
            child = p1[:cp] + p2[cp:]
            children.append(child)

        population.extend(children)

        # Mutasi
        for ind in population:
            if random.random() < mutation_rate:
                ind[0] = random.randint(8, 64)  # mutate units

    # Best individual
    final_scores = [fitness_lstm(ind) for ind in population]
    best_idx = np.argmin(final_scores)
    best = population[best_idx]

    best_units = int(best[0])
    best_dropout = float(best[1])
    best_lr = float(best[2])
    best_batch = int(best[3])

    best_model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=best_units,
        dropout=best_dropout,
        lr=best_lr
    )

    history = best_model.fit(
        X_train, y_train,
        epochs=BASE_EPOCHS,
        batch_size=best_batch,
        validation_split=0.2,
        verbose=0
    )

    y_pred_scaled = best_model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()

    best_mape = mape(y_true, y_pred)

    best_params = {
        "units": best_units,
        "dropout": best_dropout,
        "lr": best_lr,
        "batch_size": best_batch
    }

    return best_model, history, best_mape, y_pred, y_true, best_params


# =========================
# SESSION STATE
# =========================
if "trained" not in st.session_state:
    st.session_state.trained = False

# =========================
# SECTION 1: PROSES DATA
# =========================
if section == "Proses Data":
    st.header("Tahap 1: Proses Data")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Preview Data Awal")
        st.dataframe(df_raw.head(10), height=200)
        st.dataframe(df.describe(), height=200)

    with col2:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe(), height=200)

    st.subheader("Visualisasi Close Price")
    fig, ax = plt.subplots(figsize=(6, 3))  # lebih kecil
    ax.plot(df.index, df["Close"])
    ax.set_title("Harga Saham (Close)")
    st.pyplot(fig, use_container_width=False)

    st.markdown("### Informasi Preprocessing")
    st.write(f"Total Data: {n}")
    st.write(f"Data Training (80%): {len(train_values)}")
    st.write(f"Data Testing (20%): {len(test_values)}")
    st.write(f"Window LSTM: {window}") 
    st.write(f"Bentuk X_train: {X_train.shape}")
    st.write(f"Bentuk X_test: {X_test.shape}")

# =========================
# SECTION 2: TRAINING
# =========================
elif section == "Training & Evaluasi":
    st.header("Tahap 2: Training Model (LSTM, PSO, GA)")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        run_base = st.button("Run LSTM Baseline")
    
    with col_btn2:
        run_pso = st.button("Run LSTM-PSO")
    
    with col_btn3:
        run_ga = st.button("Run LSTM-GA")


        # BASELINE
    if run_base:
        with st.spinner("Training LSTM Baseline..."):
            st.session_state.model, \
            st.session_state.history, \
            st.session_state.mape, \
            st.session_state.y_pred, \
            st.session_state.y_true = train_baseline()

            st.session_state.trained = True
            st.session_state.model_name = "LSTM Baseline"

        st.success("Training Baseline selesai!")

    # PSO
    if run_pso:
        with st.spinner("Optimasi PSO + Training LSTM..."):
            model, history, mape_val, y_pred, y_true, params = train_pso()

            st.session_state.model = model
            st.session_state.history = history
            st.session_state.mape = mape_val
            st.session_state.y_pred = y_pred
            st.session_state.y_true = y_true
            st.session_state.trained = True
            st.session_state.model_name = "LSTM-PSO"
            st.session_state.best_params = params

        st.success("Training LSTM-PSO selesai!")

    # GA
    if run_ga:
        with st.spinner("Optimasi GA + Training LSTM..."):
            model, history, mape_val, y_pred, y_true, params = train_ga()

            st.session_state.model = model
            st.session_state.history = history
            st.session_state.mape = mape_val
            st.session_state.y_pred = y_pred
            st.session_state.y_true = y_true
            st.session_state.trained = True
            st.session_state.model_name = "LSTM-GA"
            st.session_state.best_params = params

        st.success("Training LSTM-GA selesai!")

        st.subheader("Model yang Digunakan")
        st.write(st.session_state.get("model_name", "LSTM Baseline"))

        if "best_params" in st.session_state:
            st.subheader("Best Hyperparameter (Optimasi)")
            st.json(st.session_state.best_params)


# =========================
# SECTION 3: FORECAST
# =========================
elif section == "Forecast":
    st.header("Tahap 3: Forecast Future")

    if not st.session_state.trained:
        st.warning("Jalankan training terlebih dahulu.")
    else:
        future_days = st.slider("Forecast Horizon (Hari)", 5, 30, 7)

        last_window = X_test[-1].copy()
        future_preds = []

        for _ in range(future_days):
            pred = st.session_state.model.predict(
                last_window.reshape(1, last_window.shape[0], last_window.shape[1]),
                verbose=0
            )
            future_preds.append(pred[0, 0])
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred

        future_preds = scaler.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).flatten()

        st.subheader("Grafik Forecast")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(future_preds, label="Forecast")
        ax3.set_title("Forecast")
        ax3.legend()
        st.pyplot(fig3, use_container_width=False)

