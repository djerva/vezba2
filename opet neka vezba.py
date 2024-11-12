import numpy as np            # Za numeričke operacije
import pandas as pd           # Za rad sa podacima
import matplotlib.pyplot as plt  # Za grafički prikaz podataka
from sklearn.preprocessing import MinMaxScaler  # Za normalizaciju podataka
from tensorflow.keras.models import Sequential  # Za izgradnju modela
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Za slojeve modela

# Generišemo sintetičke podatke (sinusni talas sa šumom)
np.random.seed(0)  # Postavljanje semena za reprodukciju
data_length = 1000  # Broj podataka
time = np.arange(data_length)  # Vremenska os
data = np.sin(0.01 * time) + np.random.normal(scale=0.5, size=data_length)  # Sinusni talas sa šumom

# Normalizacija podataka
data = data.reshape(-1, 1)  # Preoblikovanje u 2D oblik (1000, 1)
scaler = MinMaxScaler(feature_range=(0, 1))  # Postavljanje skale između 0 i 1
scaled_data = scaler.fit_transform(data)  # Normalizujemo podatke

# Kreiranje ulaza i izlaza
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])  # Ulaz
        Y.append(data[i + time_step, 0])      # Izlaz
    return np.array(X), np.array(Y)

time_step = 10  # Koristićemo 10 prethodnih koraka
X, Y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Preoblikovanje za LSTM

# Izgradnja modela
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))  # Prvi LSTM sloj
model.add(Dropout(0.2))  # Dropout sloj za regularizaciju
model.add(LSTM(50, return_sequences=False))  # Drugi LSTM sloj
model.add(Dropout(0.2))
model.add(Dense(1))  # Izlazni sloj

model.compile(optimizer='adam', loss='mean_squared_error')  # Koristimo Adam optimizator i MSE
model.fit(X, Y, epochs=50, batch_size=32)  # Treniramo model

predictions = model.predict(X)  # Predikcija
predictions = scaler.inverse_transform(predictions)  # Inverzna transformacija da dobijemo originalne vrednosti

# Prikaz rezultata
plt.figure(figsize=(14, 5))  # Veličina grafika
plt.plot(data, label='Original Data')  # Originalni podaci
plt.plot(np.arange(time_step, len(predictions) + time_step), predictions, label='Predictions', color='red')  # Predikcije
plt.title('LSTM Model Predictions')  # Naslov
plt.xlabel('Time Step')  # Oznaka X ose
plt.ylabel('Value')  # Oznaka Y ose
plt.legend()  # Prikaz legendi
plt.show()  # Prikaz grafika
