# Restructured version of the standalone LightGBM model + scaler app
# using the Multi-Model Air Pollution Classifier layout

import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import plotly.express as px
from PIL import Image
from datetime import datetime

st.set_page_config(page_title="Air Pollution Classifier", page_icon="🌫️", layout="wide")

st.title("🌫️ Air Pollution Level Classifier")

st.markdown("""
<div style="font-size:16px; line-height:1.6;">
    This app predicts pollution level categories based on meteorological conditions and pollutant levels.

</div>
""", unsafe_allow_html=True)

# --- Load Single Model and Scaler ---
@st.cache_data
def load_model_and_scaler():
    model_path = "LightGBM.pkl"
    scaler_path = "StandardScalar.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("\u26a0\ufe0f Model or Scaler file not found! Please upload 'LightGBM.pkl' and 'StandardScalar.pkl'.")
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model_and_scaler()
if model is None or scaler is None:
    st.stop()

# --- Sidebar Navigation ---

view_option = st.sidebar.radio("Select View", ["Introduction", "EDA", "Metrics Comparison", "Modelling & Prediction", "Feature Importance", "SHAP",'Confusion Matrix', 'HeatMap'])

# --- Introduction ---
if view_option == "Introduction":
    st.subheader("📌 Project Introduction")
    st.markdown("""
    This project predicts air pollution levels using meteorological and pollutant-based inputs. 
    Multiple models were compared, with **LightGBM** emerging as the best performer based on AUC and F1-score.
    The app demonstrates model insights, evaluation, and real-time predictions based on user input.
    """)

elif view_option == "Metrics Comparison":
    st.subheader("🤖 Model Comparison and Selection")

    st.markdown("""
    The following models were trained and evaluated:
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Random Forest
    - AdaBoost
    - CatBoost
    - LightGBM (Selected)
    """)

    col1, col2 = st.columns([1.5, 1])  # Adjust column width ratios as needed

    with col1:
        st.markdown("""
        #### 📈 Model Comparison
        | Model               | Train | Test  | Precision | Recall | F1     | AUC   |
        |--------------------|-------|-------|-----------|--------|--------|-------|
        | Logistic Regression| 0.820 | 0.818 | 0.815     | 0.818  | 0.816  | 0.934 |
        | **LightGBM**       | 0.894 | 0.866 | 0.866     | 0.866  | 0.866  | 0.966 |
        | KNN                | 1.000 | 0.837 | 0.835     | 0.837  | 0.836  | 0.944 |
        | Random Forest      | 0.875 | 0.853 | 0.850     | 0.853  | 0.851  | 0.958 |
        | AdaBoost           | 0.787 | 0.789 | 0.780     | 0.789  | 0.781  | 0.915 |
        | CatBoost           | 0.853 | 0.850 | 0.848     | 0.850  | 0.849  | 0.957 |
        """)

        st.markdown("""
        **Conclusion:** LightGBM was selected for deployment as it demonstrated the most balanced and robust performance—
        achieving the highest AUC score alongside consistently strong precision, recall, and F1-scores across classes.
        This indicates both accurate and reliable predictions on unseen data.
        """)

    with col2:
        model_Compare_path = "Model_Comparison.png"
        if os.path.exists(model_Compare_path):
            st.image(Image.open(model_Compare_path), caption="Model Comparison - Test Accuracy", use_container_width=True)
        else:
            st.warning("⚠️ Model comparison image not found.")
    
# --- Feature Names ---
station_options = ["Changping", "Dingling", "Dongsi", "Guanyuan"]
wind_options = ['N', 'E', 'ENE', 'ESE', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW']  # Include 'N' for UI only  # 'N' is UI-only  # 'N' is UI-only

station_features = [f'station_{s}' for s in station_options]
wind_features = [f'wd_{w}' for w in wind_options if w != 'N']  # Exclude 'N' from encoding
station_wd_features = station_features + wind_features

numeric_features = [
    'month', 'is_night', 'Rain_Flag', 'CO_NO2_ratio',
    'PM10_log', 'CO_log', 'O3_log', 'SO2_log', 'NO2_log',
    'PRES_log', 'temp_dewp_diff_log', 'inverse_wind_log'
]

# Hardcoded area type mapping (not used in model)
station_to_area_type = {
    'Changping': 'Suburban',
    'Dingling': 'Industrial',  # Example default for testing
    'Dongsi': 'Urban',
    'Guanyuan': 'Rural'  # Assign as needed
}

final_feature_names = station_wd_features + numeric_features

# --- Prediction ---
if view_option == "Modelling & Prediction":
    st.subheader("📅 Input Environmental Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        pm10 = st.number_input("PM10 (μg/m³)", value=100.0)
        so2 = st.number_input("SO₂ (μg/m³)", value=10.0)
        no2 = st.number_input("NO₂ (μg/m³)", value=20.0)
        co = st.number_input("CO (mg/m³)", value=1.0)
        o3 = st.number_input("O₃ (μg/m³)", value=30.0)
        pres = st.number_input("Pressure (hPa)", value=1000.0)

    with col2:
        temp_dewp_diff = st.number_input("Temp - Dew Point Diff (°C)", value=20.0)
        inverse_wind = st.number_input("Inverse Wind Speed (1/WSPM)", value=1.0)
        co_no2_ratio = st.number_input("CO / NO₂ Ratio", value=1.0)
        month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: datetime(1900, x, 1).strftime('%B'))

    with col3:
        station = st.selectbox("Station", station_options)
        wd = st.selectbox("Wind Direction", wind_options)
        is_night = st.radio("Night Time?", [0, 1], index=0)
        rain_flag = st.radio("Did it Rain?", [0, 1], index=0)
        area_type = station_to_area_type.get(station, 'Unknown')
        st.markdown(f"**Mapped Area Type:** `{area_type}`")

    # Construct Input
    user_input = {feat: 0 for feat in final_feature_names}
    user_input[f'station_{station}'] = 1
    if wd != 'N':
        user_input[f'wd_{wd}'] = 1  # Encode only if not 'N'

    user_input['PM10_log'] = np.log1p(pm10)
    user_input['SO2_log'] = np.log1p(so2)
    user_input['NO2_log'] = np.log1p(no2)
    user_input['CO_log'] = np.log1p(co)
    user_input['O3_log'] = np.log1p(o3)
    user_input['PRES_log'] = np.log1p(pres)
    user_input['temp_dewp_diff_log'] = np.log1p(temp_dewp_diff)
    user_input['inverse_wind_log'] = np.log1p(inverse_wind)
    user_input['CO_NO2_ratio'] = co_no2_ratio
    user_input['month'] = month
    user_input['is_night'] = is_night
    user_input['Rain_Flag'] = rain_flag

    X_cat = np.array([[user_input[feat] for feat in station_wd_features]])
    X_num = np.array([[user_input[feat] for feat in numeric_features]])
    X_num_scaled = scaler.transform(X_num)
    X_input = np.concatenate([X_cat, X_num_scaled], axis=1)

    st.markdown("---")

    if st.button("🌫️ Predict Pollution Level"):
        try:
            pred_proba = model.predict_proba(X_input)[0]
            pred_class = np.argmax(pred_proba)

            class_map = {0: "High", 1: "Low", 2: "Moderate"}
            st.success(f"🌟 Predicted Pollution Level: **{class_map[pred_class]}**")

            prob_df = pd.DataFrame({
                "Pollution Level": ["High", "Low", "Moderate"],
                "Probability (%)": pred_proba * 100
            })

            fig = px.bar(
                prob_df, x='Pollution Level', y='Probability (%)',
                text='Probability (%)', color='Pollution Level',
                color_discrete_sequence=px.colors.qualitative.Dark2
            )
            fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
            fig.update_layout(
                yaxis=dict(range=[0, 100]), showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"\u26a0\ufe0f Prediction failed: {e}")

# --- Feature Importance ---
# --- Feature Importance ---
if view_option == "Feature Importance":
    st.subheader("📊 Top Features Impacting Pollution Prediction")
    st.markdown("""
    This bar chart visualizes the top 20 most influential features used by the LightGBM model. 
    Notably, **PM10 concentration**, **temperature-dew point difference**, and **SO₂ levels** are among the most dominant factors contributing to pollution classification.
    
    These features represent critical environmental indicators:
    - **PM10_log**: Reflects coarse particulate matter levels, the most impactful predictor.
    - **temp_dewp_diff_log**: Indicates atmospheric stability, affecting pollution dispersion.
    - **SO2_log** and **O3_log**: Represent gaseous pollutants known to affect air quality.
    - **Wind direction** and **station location** features rank lower, implying regional and directional influences are less significant than direct pollutant measures.

    This insight helps prioritize key environmental variables for both monitoring and model explainability.
    """)
    feature_importance_path = "feature_importance.png"
    if os.path.exists(feature_importance_path):
        st.image(Image.open(feature_importance_path), caption="Feature Importance - LightGBM", width=700)
    else:
        st.warning("\u26a0\ufe0f Feature importance image not found.")


# --- SHAP ---
elif view_option == "SHAP":
    st.subheader("🧠 SHAP Summary Visualization")
    st.markdown("""
    The SHAP summary plot shown below provides a **global explanation** of the LightGBM model by quantifying each feature’s average contribution to predictions across all pollution level classes.

    - 📊 **Bar Length**: Represents the mean absolute SHAP value, indicating the **average influence** of a feature on model output.
    - 🎨 **Colors**: Different colors denote the **class-wise impact** on the three pollution categories:
        - 🟦 **Class 0 (High Pollution)**
        - 🟪 **Class 1 (Low Pollution)**
        - 🟩 **Class 2 (Moderate Pollution)**

    ### Key Interpretations:
    - ✅ **PM10_log** emerges as the most influential feature, reflecting the central role of particulate matter in air quality degradation.
    - ✅ **CO_log**, **temperature-dew point difference**, and **SO₂ levels** follow closely, underscoring the model’s emphasis on **pollutant concentration and atmospheric conditions**.
    - ⚠️ Features such as **station location** and **wind direction** had relatively lower impact, suggesting that temporal and chemical factors dominate over spatial ones.

    ### Why This Matters:
    This visualization not only boosts trust in the model but also **aids domain experts** in validating that the model aligns with established environmental science—confirming that pollutant-driven metrics are primary determinants of air quality classification.
    """)
    
    shap_path = "shap_LightGBM.png"
    if os.path.exists(shap_path):
        st.image(Image.open(shap_path), caption="SHAP Summary - LightGBM", width=700)
    else:
        st.warning("⚠️ SHAP visualization not found.")

# --- EDA ---
elif view_option == "EDA":
    st.subheader("\U0001f50d Exploratory Data Analysis")
    eda_path = "eda_summary.png"
    if os.path.exists(eda_path):
        st.image(Image.open(eda_path), caption="Overall EDA Summary", use_column_width=True)
    else:
        st.warning("\u26a0\ufe0f EDA summary image not found.")

elif view_option == "Confusion Matrix":
    st.subheader("🔍 Confusion Matrix - LightGBM (Test Data)")
    st.markdown("""
    The confusion matrix below showcases the predictive strength of the LightGBM model across the three pollution categories.

    - ✅ **Class 0 (High Pollution)**: Achieved an impressive classification accuracy with 4,122 correct predictions and minimal misclassifications.
    - ✅ **Class 1 (Low Pollution)**: Also demonstrated strong predictive capability with 4,087 correctly identified cases, reinforcing the model's reliability.
    - 🔄 **Class 2 (Moderate Pollution)**: While slightly more prone to misclassification due to its intermediate nature, 2,188 instances were accurately classified, underscoring the model’s ability to handle subtle category overlaps.

    Overall, the model generalizes effectively across the classes, handling imbalanced data with commendable consistency. Misclassifications were predominantly between neighboring pollution levels, which is expected due to the inherent similarity in environmental patterns.

    This matrix confirms that the deployed LightGBM model is well-calibrated, with high predictive precision and strong recall performance—validating its selection for real-world deployment.
    """)
    confusion_matrix_path = "confusion_matrix.png"
    if os.path.exists(confusion_matrix_path):
        st.image(Image.open(confusion_matrix_path), caption="Confusion Matrix - LightGBM", width=700)
    else:
        st.warning("⚠️ Confusion matrix image not found.") 


if view_option == "HeatMap":
    st.subheader("🔄 Correlation Matrix of Air Quality Variables")
    st.markdown("""
    This heatmap visualizes the correlation between different features in our air pollution dataset. 
    The color intensity indicates the strength of correlation, with deep red showing strong positive correlations and deep blue showing strong negative correlations.
    
    Key correlation patterns revealed:
    - **Pollutant Relationships**: Strong positive correlations exist between related pollutants like CO, NO2, SO2, and O3, suggesting these pollutants often increase together.
    - **PM10 Correlations**: PM10 shows significant positive correlation with SO2 (0.47) and NO2 (0.59), indicating common emission sources.
    - **Meteorological Influences**: Temperature-dew point difference correlates negatively with several pollutants, showing how atmospheric conditions affect pollution levels.
    - **Station Independence**: Low correlations between different monitoring stations (visible in the upper left quadrant), suggesting localized pollution patterns.
    - **Wind Direction Effects**: Wind directions show minimal correlation with pollutant levels, with correlation coefficients generally between -0.1 and 0.1.

    **Tree-Based Model Advantage**: 
    While this correlation matrix reveals significant multicollinearity among certain pollutants, our primary models (LightGBM, XGBoost) are tree-based algorithms that are naturally robust against collinearity issues. Unlike linear models where multicollinearity can cause unstable coefficients and inflated variance, tree-based models make splits based on information gain rather than linear combinations of features. This allowed us to retain all these interrelated variables without dimensionality reduction, preserving the complex environmental interactions crucial for accurate pollution prediction.
    
    **For Linear or Distance-Based Models**:
    If switching to linear regression, logistic regression, SVM, or k-NN algorithms, the following variables should be addressed due to high collinearity:
    
    1. **Gas Pollutant Group** (r > 0.4):
       - Keep only one of: CO_log, NO2_log, SO2_log, O3_log (possibly NO2_log as most representative)
       - Drop CO_NO2_ratio as it's derived from already correlated variables
       
    2. **Wind & Temperature Variables**:
       - The inverse_wind_log and temp_dewp_diff_log show moderate correlation (r ≈ -0.42)
       - Keep temp_dewp_diff_log and remove inverse_wind_log
       
    3. **Categorical One-Hot Encodings**:
       - For station_* variables: drop one station as the reference level
       - For wind direction (wd_*) variables: drop one direction as the reference level
       
    A reduced feature set for linear/distance models might include: PM10_log, NO2_log, temp_dewp_diff_log, Rain_Flag, PRES_log, month, night_time, and select categorical dummies.
    """)
    heatmap_path = "heatmap.png"
    if os.path.exists(heatmap_path):
        st.image(Image.open(heatmap_path), caption="Correlation Matrix of Air Quality Variables", width=700)
    else:
        st.warning("\u26a0\ufe0f Correlation heatmap image not found.")