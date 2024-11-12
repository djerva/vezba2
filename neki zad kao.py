import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Generišite sintetičke podatke
np.random.seed(0)
data_length = 1000
time = np.arange(data_length)
data = np.sin(0.01 * time) + np.random.normal(scale=0.5, size=data_length)  # Sinusni talas sa šumom

# 2. Pripremite podatke
data = data.reshape(-1, 1)  # Preoblikujte podatke
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)  # Normalizujte podatke

# 3. Kreirajte ulazne i izlazne podatke za model
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10  # Broj prethodnih koraka koje model koristi
X, Y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Preoblikujte X za LSTM

# 4. Izgradite LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))  # Dodajte Dropout za regularizaciju
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # Izlazni sloj

# 5. Kompajlirajte model
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Trenirajte model
model.fit(X, Y, epochs=50, batch_size=32)

# 7. Predikcija
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)  # Inverzna transformacija da se vrati u originalne vrednosti

# 8. Prikaz rezultata
plt.figure(figsize=(14, 5))
plt.plot(data, label='Original Data')
plt.plot(np.arange(time_step, len(predictions) + time_step), predictions, label='Predictions', color='red')
plt.title('LSTM Model Predictions')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
