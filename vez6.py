import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import pandas as pd

# Učitavanje podataka iz vašeg .csv fajla
data = pd.read_csv('airline-passengers.csv')  # Zamenite sa pravom putanjom do fajla

# Pretpostavimo da je kolona sa vrednostima koje želite da predvidite nazvana 'value_column'
data = data['Passengers'].values  # Zamenite 'value_column' sa nazivom svoje kolone


# Normalizacija podataka
data = data.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Kreiranje ulaza i izlaza
X, Y = [], []
for i in range(len(scaled_data) - 10):
    X.append(scaled_data[i:i + 10])
    Y.append(scaled_data[i + 10])
X, Y = np.array(X), np.array(Y)

# Preoblikovanje za LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Izgradnja modela
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Treniranje modela
model.fit(X, Y, epochs=10, batch_size=32)

# Predikcija
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Prikaz rezultata
plt.figure(figsize=(14, 5))
plt.plot(data, label='Original Data')
plt.plot(np.arange(10, len(predictions) + 10), predictions, label='Predictions', color='red')
plt.title('LSTM Model Predictions')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.legend()
plt.show()