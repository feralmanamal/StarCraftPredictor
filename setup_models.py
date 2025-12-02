import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def train_and_save_artifacts():
    print("--- Starting Setup Process ---")
    
    # 1. Load Data
    print("Loading dataset...")
    try:
        df = pd.read_csv('starcraft.csv')
    except FileNotFoundError:
        print("Error: 'starcraft.csv' not found. Please ensure it is in the same directory.")
        return

    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # Convert columns to numeric
    for col in ['Age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    # 2. Feature Engineering
    # We load all potential raw features needed for calculations
    features_to_load = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'NumberOfPACs',]
    
    X = df[features_to_load].copy()
    y = df['LeagueIndex']

    # Apply Transformation: EfficiencyIndex = APM / ActionLatency
    X['EfficiencyIndex'] = X['APM'] / X['ActionLatency']
    
    # Drop APM and ActionLatency as they are replaced by EfficiencyIndex
    X = X.drop(columns=['APM', 'ActionLatency'])
    
    # Final Feature Order: ['SelectByHotkeys', 'GapBetweenPACs', 'NumberOfPACs', 'EfficiencyIndex']
    print(f"Training with features: {X.columns.tolist()}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scale Data
    print("Fitting and saving scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler_ver1.joblib')

    # Save test data for App EDA
    X_test.to_csv('X_test_processed.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    metrics_store = {}

    # --- MODEL 1: RANDOM FOREST ---
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_features='sqrt')
    rf_model.fit(X_train_scaled, y_train)
    
    rf_pred = rf_model.predict(X_test_scaled)
    metrics_store['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2': r2_score(y_test, rf_pred)
    }
    joblib.dump(rf_model, 'rf_model.joblib')



    # --- MODEL 2: FNN (Neural Network) ---
    print("Training Neural Network...")
    fnn_model = Sequential()
    fnn_model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    fnn_model.add(Dropout(0.2))
    fnn_model.add(Dense(32, activation='relu'))
    fnn_model.add(Dropout(0.2))
    fnn_model.add(Dense(1, activation='linear'))
    
    fnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    fnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=0, validation_split=0.2)
    
    fnn_pred = fnn_model.predict(X_test_scaled).flatten()
    metrics_store['Neural Network'] = {
        'MAE': mean_absolute_error(y_test, fnn_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, fnn_pred)),
        'R2': r2_score(y_test, fnn_pred)
    }
    fnn_model.save('fnn_model.h5')

    # Save metrics
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_store, f)

    print("\nSUCCESS: All models and metrics (including RMSE) saved.")
    print("\n\n\nYou may now run 'starcraft_rank_app.py' to use the models in the app.\n\n\n")

if __name__ == "__main__":
    print("You are only required to run this code once, then the models are setup to run the the app")
    print("\n\n\nPlease allow for 30 seconds for model training to complete...\n\n\n\n\n")
    train_and_save_artifacts()