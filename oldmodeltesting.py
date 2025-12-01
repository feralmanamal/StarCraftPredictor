import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
import matplotlib.pyplot as plt

# imports the dataset and drops rows with missing values (loss of ~30 rows)
df = pd.read_csv('starcraft.csv')
df = df.replace('?', np.nan)
df = df.dropna()

features = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'HoursPerWeek', 'NumberOfPACs']

x = df[features]
y = df['LeagueIndex']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = Sequential()
model.add(Dense(64, input_dim=x_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2)) # randomly disables 20% of neurons to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) # randomly disables 20% of neurons to prevent overfitting
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(x_scaled, y, epochs=50, batch_size=10, validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

    # Set up the figure for side-by-side plots
plt.figure(figsize=(12, 5))

    # --- Plot 1: Loss (MSE) ---
plt.subplot(1, 2, 1)
plt.plot(hist['epoch'], hist['loss'], label='Training Loss (MSE)')
plt.plot(hist['epoch'], hist['val_loss'], label='Validation Loss (MSE)')
plt.title('Training and Validation Loss (MSE) Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

    # --- Plot 2: MAE ---
plt.subplot(1, 2, 2)
plt.plot(hist['epoch'], hist['mae'], label='Training MAE')
plt.plot(hist['epoch'], hist['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE Over Time')
plt.xlabel('Epoch')
plt.ylabel('MAE (League Index Error)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()