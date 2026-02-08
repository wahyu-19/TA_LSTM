import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.preprocessing import make_sequences, scale_split
from utils.metrics import mape
from models.baseline import build_baseline

st.set_page_config(layout="wide")

st.sidebar.title("Menu")
menu = st.sidebar.radio("",[
    "Informasi Data",
    "In Depth Analysis",
    "Hasil Forecast"
])

symbol = st.sidebar.text_input("Ticker","SIDO.JK")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load(symbol):
    df = yf.download(symbol,start="2019-01-01")
    return df[['Close']]

df = load(symbol)

values = df.values
train_scaled,test_scaled,scaler = scale_split(values)

X_train,y_train = make_sequences(train_scaled)
X_test,y_test = make_sequences(test_scaled)

# ===============================
# TRAIN BASELINE
# ===============================
model = build_baseline((X_train.shape[1],X_train.shape[2]))
history = model.fit(X_train,y_train,epochs=10,validation_split=0.2,verbose=0)

pred = scaler.inverse_transform(model.predict(X_test))
true = scaler.inverse_transform(y_test)

mape_val = mape(true,pred)

# ===============================
# MENU 1
# ===============================
if menu == "Informasi Data":

    st.title("Informasi Data")

    st.write(df.describe())

    fig,ax = plt.subplots()
    ax.plot(df['Close'])
    st.pyplot(fig)

# ===============================
# MENU 2
# ===============================
if menu == "In Depth Analysis":

    st.title("Model Analysis")

    st.write("MAPE:",mape_val)

    fig,ax = plt.subplots()
    ax.plot(history.history['loss'],label="train")
    ax.plot(history.history['val_loss'],label="val")
    ax.legend()
    st.pyplot(fig)

# ===============================
# MENU 3
# ===============================
if menu == "Hasil Forecast":

    horizon = st.slider("Forecast horizon",1,30,5)

    last_seq = test_scaled[-1].reshape(1,1,1)
    preds=[]

    for i in range(horizon):
        p = model.predict(last_seq)
        preds.append(p[0,0])
        last_seq = p.reshape(1,1,1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))

    future_dates = pd.date_range(df.index[-1],periods=horizon+1)[1:]

    fig,ax = plt.subplots()
    ax.plot(df.index,df['Close'])
    ax.plot(future_dates,preds)
    st.pyplot(fig)
