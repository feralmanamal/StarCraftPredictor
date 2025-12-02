# StarCraft Rank Predictor App
# How to use:
# 1. Set command prompt directory to this package. 
# 2. Ensure all libraries are downloaded
# 3. run: python -m streamlit run starcraft_rank_app.py


#Definitions:
# PACs = Perception Action Cycle (s) (speed of a reaction to an event per event)
# APM = Actions per miniute
# GapsBetweenPACs = Time between perception action evenets
# SelectByHotkeys = How often do you use hotkeys to selection something (frequency)
# NumberOfPACs = Number of PACs per timestamp 


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress specific warnings if using older Streamlit versions
# st.set_option('deprecation.showPyplotGlobalUse', False) 
# ^ Commented out as requested by error log, fixed by explicit fig passing below

st.set_page_config(page_title="StarCraft Rank Predictor", layout="wide")

# League Mapping for better UI
LEAGUE_MAP = {
    1: "Bronze", 2: "Silver", 3: "Gold", 4: "Platinum",
    5: "Diamond", 6: "Master", 7: "GrandMaster", 8: "Professional"
}

# Features used for EDA visualization (Raw data)
EDA_FEATURES = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'NumberOfPACs']
TARGET = 'LeagueIndex'

@st.cache_data
def load_data_for_eda():
    """Loads the raw dataset specifically for EDA visualizations."""
    try:
        df = pd.read_csv('starcraft.csv')
        df = df.replace('?', np.nan)
        # Convert numeric columns
        cols_to_numeric = ['Age']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        df = df.dropna()
        return df
    except FileNotFoundError:
        st.error("Error: 'starcraft.csv' not found. EDA tabs may be empty.")
        return None

@st.cache_resource
def load_models_and_scaler():
    """Loads the pre-trained models and the specific scaler."""
    try:
        # Load the scaler
        scaler = joblib.load('scaler_ver1.joblib')
        
        # Load models
        rf_model = joblib.load('rf_model.joblib')
        
        # Load Keras model with explicit custom objects to handle 'mse' metric error
        fnn_model = tf.keras.models.load_model('fnn_model.h5', 
                                               custom_objects={'mse': tf.keras.losses.MeanSquaredError(), 
                                                             'mae': tf.keras.metrics.MeanAbsoluteError()})
        
        # Load metrics
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
             
        return scaler, rf_model, fnn_model, metrics
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Please run 'setup_models.py' first to generate them!")
        return None, None, None, None, None

def preprocess_user_input(user_data, scaler):
    """
    Applies the EXACT feature engineering from your scripts:
    1. Calculates EfficiencyIndex = APM / ActionLatency
    2. Drops APM and ActionLatency
    3. Scales the remaining features
    """
    df = user_data.copy()
    
    # Feature Engineering
    df['EfficiencyIndex'] = df['APM'] / df['ActionLatency']
    
    # Select and Order columns exactly as the model expects
    cols_for_model = ['SelectByHotkeys', 'GapBetweenPACs', 'NumberOfPACs', 'EfficiencyIndex']
    df_processed = df[cols_for_model]
    
    # Scale using the loaded scaler
    df_scaled = scaler.transform(df_processed)
    return df_scaled

def main():
    st.title("StarCraft Rank Predictor")
    st.markdown("""
    This application predicts a player's **League Rank** based on their in-game statistics using a **Random Forest Regression** model.
    It includes Exploratory Data Analysis (EDA) and Model Performance metrics.
    """)

    # Load resources
    scaler, rf_model, fnn_model, metrics = load_models_and_scaler()
    df_eda = load_data_for_eda()

    if not scaler:
        st.stop()

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Player Stats Input")
    st.sidebar.markdown("Enter your gameplay statistics below:")
    
    with st.sidebar.form("prediction_form"):
        apm = st.number_input("APM (Actions Per Minute)", min_value=0.0, max_value=600.0, value=100.0)
        action_latency = st.number_input("Action Latency (ms)", min_value=1.0, max_value=200.0, value=60.0)
        gap_pacs = st.number_input("Gap Between PACs", min_value=0.0, max_value=200.0, value=40.0)
        
        # High precision inputs for small values
        select_hotkeys = st.number_input("Select By Hotkeys (frequency)", 
                                        min_value=0.0, max_value=1.0, value=0.002, format="%.6f", step=0.0001)
        num_pacs = st.number_input("Number of PACs", 
                                   min_value=0.0, max_value=0.1, value=0.003, format="%.6f", step=0.0001)
        
        
        # Model Selection
        selected_model_name = st.selectbox("Select Model", ["Random Forest", "Neural Network (FNN)"])
        
        submit_button = st.form_submit_button("Predict Rank")

    # --- MAIN TABS ---
    tab1, tab2, tab3 = st.tabs([" Prediction", " Model Performance", "Data Analysis (EDA)"])

    # TAB 1: PREDICTION
    with tab1:
        st.subheader("Prediction Results")
        
        if submit_button:
            # 1. Create DataFrame from inputs
            # We still collect APM and ActionLatency to calculate EfficiencyIndex
            user_df = pd.DataFrame({
                'APM': [apm],
                'SelectByHotkeys': [select_hotkeys],
                'ActionLatency': [action_latency],
                'GapBetweenPACs': [gap_pacs],
                'NumberOfPACs': [num_pacs],
            })
            
            # 2. Preprocess (Engineer features + Scale)
            input_scaled = preprocess_user_input(user_df, scaler)
            
            # 3. Predict
            if selected_model_name == "Random Forest":
                pred = rf_model.predict(input_scaled)[0]
            else: # FNN
                pred = fnn_model.predict(input_scaled).flatten()[0]
            
            pred_rank = int(round(pred))
            pred_rank = max(1, min(8, pred_rank)) # Clamp between 1 and 8
            
            # 4. Display
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted League", value=f"{LEAGUE_MAP[pred_rank]} ({pred_rank})")
            with col2:
                st.metric(label="Raw Model Score", value=f"{pred:.4f}")
                
            if pred_rank >= 6:
                st.success("Impressive! You're playing at a high level.")
            elif pred_rank >= 3:
                st.info("Solid performance. Keep practicing!")
            else:
                st.warning("Keep working on your mechanics!")
                
        else:
            st.info("Adjust the parameters in the sidebar and click 'Predict Rank' to see your result.")

        # Raw Data Preview
        if df_eda is not None:
            with st.expander("View Raw Dataset Sample"):
                st.dataframe(df_eda.head())

    # TAB 2: MODEL PERFORMANCE
    with tab2:
        st.subheader("Model Comparison Metrics")
        st.markdown("Performance metrics calculated on the test set.")
        
        # Display Metrics in a nice table
        metrics_df = pd.DataFrame(metrics).T
        
        # Check if columns exist before highlighting to avoid errors
        available_cols = metrics_df.columns.tolist()
        min_cols = [col for col in ['MAE', 'RMSE'] if col in available_cols]
        max_cols = [col for col in ['R2'] if col in available_cols]
        
        # Safe styling
        st_df = metrics_df.style
        if min_cols:
            st_df = st_df.highlight_min(subset=min_cols, axis=0)
        if max_cols:
            st_df = st_df.highlight_max(subset=max_cols, axis=0)
            
        st.dataframe(st_df)
        
        # Feature Importance (Only available for Random Forest in this context)
        if "Random Forest" in metrics:
            st.write("---")
            st.write("**Feature Importance (Random Forest)**")
            st.caption("Shows which engineered features drive the prediction most.")
            
            # The features used in training
            trained_features = ['SelectByHotkeys', 'GapBetweenPACs', 'NumberOfPACs', 'EfficiencyIndex']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            sns.barplot(x=importances[indices], y=np.array(trained_features)[indices], ax=ax_imp, palette="viridis")
            ax_imp.set_xlabel("Importance")
            st.pyplot(fig_imp)

    # TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
    with tab3:
        st.subheader("Exploratory Data Analysis")
        
        if df_eda is not None:
            col_eda1, col_eda2 = st.columns(2)
            
            # Re-create feature engineering for visualization purposes on the full dataset
            df_eng = df_eda.copy()
            df_eng['EfficiencyIndex'] = df_eng['APM'] / df_eng['ActionLatency']
            
            # Columns to show in correlation matrix
            cols_to_show = ['SelectByHotkeys', 'GapBetweenPACs', 'NumberOfPACs', 'EfficiencyIndex', 'LeagueIndex']
            
            with col_eda1:
                st.write("**Correlation Matrix (Engineered Features)**")
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                corr = df_eng[cols_to_show].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
                st.pyplot(fig3)
                
            with col_eda2:
                st.write("**Target Distribution**")
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                sns.histplot(df_eda[TARGET], bins=8, kde=True, ax=ax4, color='purple')
                ax4.set_xlabel("League Index")
                st.pyplot(fig4)
        else:
            st.warning("Dataset not found, cannot display EDA.")

if __name__ == "__main__":
    main()