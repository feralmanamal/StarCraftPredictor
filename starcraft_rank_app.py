# ==========================================
# StarCraft Rank Predictor App
# ==========================================
# This script combines Data Cleaning, EDA, Modelling (Random Forest),
# and a Frontend Interface into a single Streamlit application.
#
# Author: Generated for DS3000 Assignment
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. CONFIGURATION & CACHING
# ==========================================
st.set_page_config(page_title="StarCraft Rank Predictor", layout="wide")

# Features selected based on model.py analysis
FEATURES = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'HoursPerWeek', 'NumberOfPACs']
TARGET = 'LeagueIndex'

# League Mapping for better UI
LEAGUE_MAP = {
    1: "Bronze", 2: "Silver", 3: "Gold", 4: "Platinum",
    5: "Diamond", 6: "Master", 7: "GrandMaster", 8: "Professional"
}

@st.cache_data
def load_data():
    """Loads and cleans the StarCraft dataset."""
    try:
        df = pd.read_csv('starcraft.csv')
        
        # Handling missing values denoted by '?'
        df = df.replace('?', np.nan)
        
        # Convert numeric columns
        cols_to_numeric = ['Age', 'HoursPerWeek', 'TotalHours']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        # Drop missing values
        df = df.dropna()
        return df
    except FileNotFoundError:
        st.error("Error: 'starcraft.csv' not found in the directory. Please make sure the file exists.")
        return None

@st.cache_resource
def train_model(df):
    """Trains the Random Forest model and returns artifacts."""
    X = df[FEATURES]
    y = df[TARGET]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model: Random Forest (Selected over Neural Net for stability on this tabular data)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_scaled)
    metrics = {
        'MAE': mean_absolute_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'R2': r2_score(y_test, predictions)
    }
    
    return model, scaler, metrics, X_test, y_test, predictions

# ==========================================
# 2. APP LAYOUT
# ==========================================
def main():
    st.title("🏆 StarCraft II Rank Predictor")
    st.markdown("""
    This application predicts a player's **League Rank** based on their in-game statistics using a **Random Forest Regression** model.
    It includes Exploratory Data Analysis (EDA) and Model Performance metrics.
    """)

    # Load Data
    df = load_data()
    if df is None:
        st.stop()

    # Train Model (Cached)
    model, scaler, metrics, X_test, y_test, predictions = train_model(df)

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("🎮 Player Stats Input")
    st.sidebar.markdown("Enter your gameplay statistics below:")
    
    with st.sidebar.form("prediction_form"):
        apm = st.number_input("APM (Actions Per Minute)", min_value=0.0, max_value=600.0, value=100.0)
        action_latency = st.number_input("Action Latency (ms)", min_value=0.0, max_value=200.0, value=60.0)
        gap_pacs = st.number_input("Gap Between PACs", min_value=0.0, max_value=200.0, value=40.0)
        hours_week = st.number_input("Hours Played Per Week", min_value=0, max_value=168, value=10)
        
        # High precision inputs for small values
        select_hotkeys = st.number_input("Select By Hotkeys (frequency)", 
                                        min_value=0.0, max_value=0.1, value=0.002, format="%.6f", step=0.0001)
        num_pacs = st.number_input("Number of PACs", 
                                   min_value=0.0, max_value=0.01, value=0.003, format="%.6f", step=0.0001)
        
        submit_button = st.form_submit_button("Predict Rank")

    # --- MAIN TABS ---
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Data Analysis (EDA)"])

    # TAB 1: PREDICTION
    with tab1:
        st.subheader("Prediction Results")
        
        if submit_button:
            # Create input dataframe matching training features exactly
            input_data = pd.DataFrame({
                'APM': [apm],
                'SelectByHotkeys': [select_hotkeys],
                'ActionLatency': [action_latency],
                'GapBetweenPACs': [gap_pacs],
                'HoursPerWeek': [hours_week],
                'NumberOfPACs': [num_pacs]
            })
            
            # Scale Input
            input_scaled = scaler.transform(input_data)
            
            # Predict
            pred_float = model.predict(input_scaled)[0]
            pred_rank = int(round(pred_float))
            pred_rank = max(1, min(8, pred_rank)) # Clamp between 1 and 8
            
            # Display
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted League", value=f"{LEAGUE_MAP[pred_rank]} ({pred_rank})")
            with col2:
                st.metric(label="Raw Model Score", value=f"{pred_float:.2f}")
                
            if pred_rank >= 6:
                st.success("Impressive! You're playing at a high level.")
            elif pred_rank >= 3:
                st.info("Solid performance. Keep practicing!")
            else:
                st.warning("Keep working on your mechanics!")
        else:
            st.info("Adjust the parameters in the sidebar and click 'Predict Rank' to see your result.")

        # Raw Data Preview
        with st.expander("View Raw Dataset Sample"):
            st.dataframe(df.head())

    # TAB 2: MODEL PERFORMANCE
    with tab2:
        st.subheader("Random Forest Model Evaluation")
        
        # Metrics Display
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score", f"{metrics['R2']:.4f}", help="Variance explained by the model (1.0 is perfect)")
        m_col2.metric("MAE", f"{metrics['MAE']:.4f}", help="Average absolute error in rank prediction")
        m_col3.metric("RMSE", f"{metrics['RMSE']:.4f}", help="Root Mean Squared Error")
        
        st.divider()
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.write("**Feature Importance**")
            st.caption("Which stats matter most for your rank?")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create figure explicitly
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=importances[indices], y=np.array(FEATURES)[indices], ax=ax, palette="viridis")
            ax.set_xlabel("Importance")
            st.pyplot(fig)
            
        with col_p2:
            st.write("**Actual vs Predicted**")
            st.caption("How close were the predictions to reality?")
            # Create figure explicitly
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(y_test, predictions, alpha=0.3, color='blue')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_xlabel("Actual Rank")
            ax2.set_ylabel("Predicted Rank")
            st.pyplot(fig2)

    # TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
    with tab3:
        st.subheader("Exploratory Data Analysis")
        
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            st.write("**Correlation Matrix**")
            # Create figure explicitly
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            corr = df[FEATURES + [TARGET]].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
            st.pyplot(fig3)
            
        with col_eda2:
            st.write("**Target Distribution**")
            # Create figure explicitly
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            sns.histplot(df[TARGET], bins=8, kde=True, ax=ax4, color='purple')
            ax4.set_xlabel("League Index")
            st.pyplot(fig4)

if __name__ == "__main__":
    main()