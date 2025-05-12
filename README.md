# Programming-with-Data-Analysis

Beijing Air Pollution Level Classifier
Air Pollution Classification
Python
Streamlit
License

📋 Overview
This interactive application classifies Beijing air pollution levels based on meteorological conditions and pollutant concentrations. Leveraging machine learning (LightGBM), the app provides real-time pollution predictions, comprehensive data visualizations, and detailed analytical insights into the factors affecting air quality across different regions of Beijing.



🚀 Quick Start
# Clone the repository
git clone https://github.com/yourusername/beijing-air-pollution-classifier.git

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run weather_app.py

🌟 Features
Interactive Pollution Predictor: Input environmental parameters and receive immediate pollution level classification
Comprehensive EDA: Explore pollution patterns across geographic regions, seasons, and area types
Model Performance Analysis: Compare various machine learning models with detailed metrics
Feature Importance Visualization: Understand the key factors driving pollution classification
SHAP Analysis: Gain insights into how the model interprets different environmental variables
Correlation Heatmap: Examine relationships between pollutants and meteorological conditions
🧪 Model Selection & Performance
After evaluating multiple classification algorithms, LightGBM was selected for its superior performance:

Model	Test Accuracy	Precision	Recall	F1	AUC
Logistic Regression	0.818	0.815	0.818	0.816	0.934
LightGBM	0.866	0.866	0.866	0.866	0.966
KNN	0.837	0.835	0.837	0.836	0.944
Random Forest	0.853	0.850	0.853	0.851	0.958
AdaBoost	0.789	0.780	0.789	0.781	0.915
CatBoost	0.850	0.848	0.850	0.849	0.957

📊 Key Findings
Geographic Disparities: Urban areas experience approximately 25% higher pollution levels compared to rural regions
Seasonal Patterns: Winter months show dramatically elevated pollution (40% higher than summer), particularly in urban centers
Pollution Distribution: Nearly equal distribution between high (37.2%) and low (37.7%) pollution episodes, with fewer moderate (25.2%) conditions
Inverse Seasonal Relationships: While primary pollutants (SO₂, NO₂, CO) peak in winter, ozone (O₃) reaches maximum levels in summer
Influential Factors: PM10 concentration, temperature-dew point difference, and SO₂ levels were identified as the strongest predictors of pollution levels

📚 Project Structure

beijing-air-pollution-classifier/
├── weather_app.py                # Main Streamlit application
├── Data_Analytics_Project.ipynb  # Jupyter notebook with analysis & modeling
├── Models/                       # Trained models
│   ├── LightGBM.pkl             # LightGBM model
│   └── StandardScalar.pkl       # Feature scaler
├── Visualizations/               # Generated visualizations
│   ├── data_distribution_overview.png
│   ├── pm25_level_distribution.png
│   ├── seasonal_pm25_by_area.png
│   └── ...
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation

🧠 Technical Approach
Data Preprocessing: Log transformations applied to highly skewed variables; categorical encoding for stations and wind directions
Feature Engineering: Created derived features like temperature-dew point difference and CO/NO₂ ratio to improve model performance
Model Training: Ensemble techniques with hyperparameter optimization via grid search
Validation Strategy: Stratified 5-fold cross-validation to ensure reliable performance across pollution categories
Implementation: Streamlit for interactive web application with Plotly for dynamic visualizations

🔮 Applications & Impact
Public Health Planning: Enable targeted health advisories based on predicted pollution levels
Urban Planning: Inform development decisions by understanding spatial pollution distribution
Policy Evaluation: Assess the effectiveness of pollution control measures over time
Personal Decision Support: Help individuals make informed choices about outdoor activities

📦 Requirements
Python 3.8+
Streamlit
NumPy
Pandas
Scikit-learn
LightGBM
Plotly
Pillow
See requirements.txt for detailed dependencies.
