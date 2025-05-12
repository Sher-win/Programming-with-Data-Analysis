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
    
    # Add project logo/image if available
    intro_image_path = "beijing_pollution_image.png"  # Update with your actual image path
    if os.path.exists(intro_image_path):
        st.image(Image.open(intro_image_path), width=600)
    
    st.markdown("""
    ## Beijing Air Pollution Classification Project
    
    This project focuses on developing an accurate classification system for air pollution levels in Beijing based on meteorological conditions and pollutant concentrations. Using extensive data from multiple monitoring stations across different area types (Urban, Suburban, Rural, Industrial, and Semi-Rural), we've created a robust machine learning solution to predict pollution severity.
    
    ### Project Goals:
    
    - **Air Quality Prediction**: Develop models that accurately classify pollution levels as Low, Moderate, or High
    - **Geographic Analysis**: Identify spatial patterns in pollution distribution across different area types
    - **Seasonal Insights**: Understand how pollution dynamics change throughout the year
    - **Model Explainability**: Identify which factors most strongly influence air pollution levels
    
    ### Key Findings:
    
    - **Geographic Disparity**: Urban and Industrial areas consistently experience higher pollution levels than Suburban zones
    - **Seasonal Patterns**: Winter shows dramatically elevated pollution levels compared to summer across all regions
    - **Inverse Relationships**: Secondary pollutants like O₃ follow opposite seasonal patterns compared to primary pollutants
    - **Predictive Success**: Our LightGBM model achieves 86.6% accuracy in classifying pollution levels
    
    ### Practical Applications:
    
    - **Public Health Planning**: Enable targeted health advisories based on predicted pollution levels
    - **Urban Planning**: Inform development decisions by understanding spatial pollution distribution
    - **Policy Evaluation**: Assess the effectiveness of pollution control measures over time
    - **Personal Decision Support**: Help individuals make informed choices about outdoor activities
    
    ### Explore the sidebar options to dive deeper into our analysis, visualizations, and interactive prediction tools.
    """)
    
    # Create columns for tech stack display
    st.markdown("### 🛠️ Technologies Used")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Data Processing**
        - Python
        - Pandas
        - NumPy
        - SciPy
        """)
    
    with col2:
        st.markdown("""
        **Modeling**
        - Scikit-learn
        - LightGBM
        - Gradient Boosting
        - Hyperparameter Tuning
        """)
    
    with col3:
        st.markdown("""
        **Visualization & Deployment**
        - Plotly
        - Matplotlib
        - Streamlit
        - Pickle
        """)
elif view_option == "EDA":
    st.subheader("\U0001f50d Exploratory Data Analysis")
    
    st.markdown("## 📊 Data Distribution Analysis")
    
    # Make image slightly smaller with width parameter
    dist_analysis_image = "data_distribution_overview.png"
    if os.path.exists(dist_analysis_image):
        st.image(Image.open(dist_analysis_image), caption="Comprehensive Distribution Analysis", width=650)
    else:
        st.warning("⚠️ Distribution analysis image not found.")
    
    # Use expander for detailed explanation
    with st.expander("👉 Click to see detailed analysis of data distributions"):
        st.markdown("""
        ### Temporal Distribution Insights
        1. **Multi-year coverage**: Dataset spans 4-5 years (2013-2017), with consistent sampling density across 2014-2016
        2. **Monthly patterns**: Cyclical distribution with slight variations, suggesting seasonal influences on data collection
        3. **Daily uniformity**: Mostly uniform across the month, with a slight spike around day 31
        4. **Hourly consistency**: Uniform sampling across all 24 hours, indicating continuous monitoring
        
        ### Pollutant Concentration Insights
        1. **PM2.5 and PM10**: Right-skewed distributions with most measurements at lower concentrations but long tails indicating pollution episodes
        2. **PM2.5 range**: Values mostly fall below 200 μg/m³ but extend to ~800 μg/m³ during extreme events
        3. **PM10 similarity**: Shows distribution pattern similar to PM2.5, with majority of values below 300 μg/m³
        4. **SO2 skewness**: Extremely right-skewed with most values below 50 μg/m³
        5. **NO2 concentration**: Primarily falls between 0-150 μg/m³ with a peak around 25-50 μg/m³
        6. **CO distribution**: Right-skewed with most values concentrated below 2000 μg/m³
        7. **O3 pattern**: Exhibits right-skewed distribution with majority of measurements below 200 μg/m³
        
        ### Meteorological Parameter Insights
        1. **Temperature**: Shows bimodal distribution, likely reflecting seasonal patterns with peaks around 0°C and 25°C
        2. **Dew point**: Also displays bimodal distribution, roughly following temperature patterns
        3. **Pressure**: Follows normal distribution centered around 1010-1015 hPa
        4. **Rainfall**: Distribution is extremely skewed with vast majority of observations showing no precipitation (0 mm)
        5. **Wind speed**: Right-skewed with predominant values below 4 m/s, suggesting generally low wind conditions
        
        ### Spatial Distribution Insights
        1. **Station locations**: Latitude/longitude distributions show distinct peaks, indicating fixed monitoring station locations
        2. **Monitoring network**: Data collected from approximately 4-5 distinct monitoring sites across Beijing
        3. **Fixed coordinates**: Monitoring stations appear to be situated at specific coordinates with minimal variation
        
        ### Modeling Considerations
        1. **Transformations needed**: Log transformations on highly skewed pollutant variables (PM2.5, PM10, SO2, O3) improve model performance
        2. **Rainfall handling**: Due to prevalence of zeros for rainfall, treating it as a binary variable (rain/no rain) is more effective
        3. **Multicollinearity potential**: Possibility of multicollinearity between related pollutants (PM2.5/PM10) and meteorological variables (TEMP/DEWP)
        """)
    
    st.markdown("---")  # Adds a visual separator between sections
    
    st.markdown("## 🥧 PM2.5 Pollution Level Distribution")
    
    pm25_pie_image = "pm25_level_distribution.png"
    if os.path.exists(pm25_pie_image):
        st.image(Image.open(pm25_pie_image), caption="PM2.5 Pollution Level Distribution", width=550)
    else:
        st.warning("⚠️ PM2.5 distribution pie chart not found.")
    
    with st.expander("👉 Click to see detailed analysis of pollution level distribution"):
        st.markdown("""
        This pie chart visualizes the overall distribution of PM2.5 air quality levels across all stations and time periods.

        ### Key Observations
        1. **Low Pollution**: Comprises the largest segment at **37.7%** of all observations
           - This indicates that good air quality conditions occur most frequently
           - Likely corresponds to values below official Chinese AQI thresholds for low concern

        2. **High Pollution**: Represents **37.2%** of the dataset
           - Nearly equal to the Low category, suggesting Beijing experiences severe pollution episodes at a frequency almost matching good air quality days
           - This substantial portion highlights the significant air quality challenges in the region

        3. **Medium Pollution**: Accounts for **25.2%** of observations
           - The smallest category, indicating pollution levels tend to polarize toward either low or high conditions
           - Transitional states occur less frequently than the extremes

        ### Health and Policy Relevance
        - The nearly equal distribution between Low and High categories suggests that Beijing residents experience both healthy and unhealthy air quality with similar frequency
        - The significant proportion of High pollution episodes (37.2%) requires the continuing need for air quality improvement policies
        - Seasonal or temporal analysis may reveal patterns in when these different categories occur
        """)

    st.markdown("---")  # Adds a visual separator between sections
    
    st.markdown("## 🏙️ PM2.5 Distribution by Area Type")
    
    area_dist_image = "pm25_by_area_type.png"
    if os.path.exists(area_dist_image):
        st.image(Image.open(area_dist_image), caption="PM2.5 Distribution Across Different Area Types", width=650)
    else:
        st.warning("⚠️ Area distribution chart not found.")
    
    with st.expander("👉 Click to see detailed analysis of area-based distribution"):
        st.markdown("""
        Looking at the bar chart showing pollution levels across different areas of Beijing, we notice fascinating patterns that challenge conventional assumptions about urban air quality.

        ### Urban vs. Rural Pollution Dynamics

        What immediately jumps out is how **Urban areas** show the highest levels of "High" category pollution (around 14,500 instances), which isn't surprising. But unexpectedly, **Rural areas** follow with similarly elevated counts of "High" pollution events. This suggests that poor air quality isn't just limited to the urban core.

        The **Suburban areas** actually recorded the highest frequency of "Low" pollution readings (approximately 14,500 counts), even exceeding Rural areas. This contradicts initial expectations and indicates that some suburban zones might benefit from favorable geographic positioning or wind patterns.

        ### Consistency in Medium Category

        Interestingly, the "Medium" pollution category remains remarkably consistent across all area types - hovering around 9,000 counts regardless of location. This stability across different environments suggests that moderate pollution levels might be influenced by regional rather than local factors.

        ### Industrial Areas: Not the Worst Offenders

        Counter to what might be anticipated, **Industrial zones** don't show the highest pollution readings. They display a more balanced profile between the three categories compared to Urban areas. This might reflect the impact of targeted emission controls in industrial zones or potentially indicate that vehicle emissions in dense urban areas are more significant contributors than industrial sources.

        ### Implications for Residents and Policy

        For Beijing residents, these findings suggest that relocating to Rural or Semi-Rural areas may not necessarily guarantee better air quality. The data shows that pollution is a regional issue that transcends simple urban-rural divides.
        """)

    st.markdown("---")  # Adds a visual separator between sections
    
    st.markdown("## 🗺️ Geographical Disadvantage in Air Pollution")
    
    geo_boxplot_image = "pm25_area_type_boxplot.png"
    if os.path.exists(geo_boxplot_image):
        st.image(Image.open(geo_boxplot_image), caption="Distribution of PM2.5 Levels by Area Type", width=650)
    else:
        st.warning("⚠️ Geographical distribution boxplot not found.")
    
    with st.expander("👉 Click to see detailed analysis of geographical disadvantage"):
        st.markdown("""
        This section focuses on analyzing how air pollution levels vary depending on the geographical classification of each monitoring station — specifically, whether it's located in an Urban, Suburban, Rural, Industrial, or Semi-Rural area.

        ### Distribution Patterns Across Area Types

        Looking at the boxplot distribution of PM2.5 across different area types, several key patterns emerge:

        1. **Urban Penalty**: Urban areas show the highest median PM2.5 concentration (~70 μg/m³), confirming that city centers bear a disproportionate pollution burden. This "urban penalty" likely results from concentrated vehicle emissions, building density, and reduced airflow in street canyons.

        2. **Semi-Rural Surprise**: Counter to what might be expected, Semi-Rural areas display the second-highest median concentration, suggesting pollution doesn't simply decrease with distance from urban centers. This may reflect topographical trapping of pollutants or proximity to peripheral industrial zones.

        3. **Outlier Episodes**: All area types experience extreme pollution episodes exceeding 600 μg/m³, with Semi-Rural and Industrial zones recording the most severe outliers (approaching 900 μg/m³). These extreme episodes appear independent of area classification.

        4. **Suburban Advantage**: Suburban areas show the lowest median PM2.5 levels, potentially benefiting from intermediate positioning that avoids both urban emission sources and regional pollution transport patterns affecting rural areas.

        These visualizations clearly show that geographical location significantly impacts air quality — with **industrial and urban areas generally facing worse conditions**, while **rural and suburban zones tend to experience better air quality**. However, even rural areas occasionally show high spikes, likely due to seasonal or wind-based factors.
        """)
        st.markdown("---")  # Adds a visual separator between sections
    
    st.markdown("## 📊 PM2.5 and PM10 Geographic Disparity")
    
    pm_geo_disparity_image = "pm_by_area_type_barchart.png"
    if os.path.exists(pm_geo_disparity_image):
        st.image(Image.open(pm_geo_disparity_image), caption="Average PM2.5 and PM10 by Area Type", width=650)
    else:
        st.warning("⚠️ PM geographic disparity chart not found.")
    
    with st.expander("👉 Click to see detailed analysis of PM2.5/PM10 geographic disparity"):
        st.markdown("""
        This bar chart reveals a clear geographic disadvantage in air quality exposure across Beijing, where residents of certain areas face significantly higher particulate matter concentrations through no choice of their own - a systemic environmental inequity embedded in the urban geography.

        ### Urban-Rural Pollution Gradient

        The most striking pattern in this visualization is the consistent decrease in both PM2.5 and PM10 concentrations as we move from Urban/Industrial to Rural areas:

        1. **Urban-Industrial Dominance**: Urban (approx 85 μg/m³) and Industrial (approx 82 μg/m³) areas show the highest average PM2.5 concentrations, exceeding the Rural average by approximately 15-18 μg/m³. This represents roughly 25% higher exposure for city dwellers compared to rural residents.

        2. **PM10 Pattern Confirmation**: The PM10 data reinforces this spatial pattern, with Urban and Industrial zones experiencing concentrations above 110 μg/m³, while Rural areas average around 90 μg/m³.

        3. **Semi-Rural Transition**: Semi-Rural areas display intermediate pollution levels (~79 μg/m³ for PM2.5), functioning as a transition zone between urban and rural environments.

        4. **Suburban Advantage**: Suburban areas show lower PM2.5 levels (~70 μg/m³) than their Semi-Rural counterparts, possibly benefiting from targeted environmental protection measures or favorable topographical positioning.

        ### PM2.5/PM10 Ratio Insights

        An interesting secondary observation is the PM2.5/PM10 ratio across different area types:

        - Rural areas show the lowest ratio (~0.77), suggesting a higher proportion of coarse particles likely from natural dust and agricultural activities
        - Urban areas display the highest ratio (~0.79), indicating a greater fraction of fine particles typical of combustion sources like vehicles and heating

        This aligns with findings from (Wang and Zhang, 2020), who noted that "the ratio of PM2.5/PM10 in urban Beijing was consistently higher than in surrounding areas, reflecting the predominance of fine particle emissions from anthropogenic sources in dense urban environments."

        ### Implications for Health and Policy

        These findings have significant public health implications. The average PM2.5 concentration exceeds 70 μg/m³ across all area types - far above the WHO guideline of 5 μg/m³. However, the ~18 μg/m³ difference between Urban and Rural areas translates to approximately 10-15% higher long-term mortality risk for urban residents based on established concentration-response functions.

        As noted by (Hu et al., 2023), "an increase of 10 μg/m³ in PM2.5 was associated with a 7.3% increase in all-cause mortality across Chinese cities." This suggests that the geographic disparity observed here has tangible health consequences for Beijing's urban population.

        (Xu et al., 2019) further emphasized that "residential proximity to industrial zones in Beijing was associated with significantly elevated PM2.5 exposure and corresponding increases in respiratory hospitalizations, creating an environmental justice concern." Our visualization provides quantitative support for this environmental inequality across Beijing's diverse geographic zones.

        ---

        **References:**
        - Hu, J. et al. (2023) 'Long-Term Exposure to PM2.5 and Mortality: A Cohort Study in China', Toxics, 11(9), p. 727.
        - Xu, Y. et al. (2019) 'Unraveling environmental justice in ambient PM2.5 exposure in Beijing: A big data approach', Computers, Environment and Urban Systems, 75, pp. 12–21.
        - Wang, X. and Zhang, R. (2020) 'Effects of atmospheric circulations on the interannual variation in PM2.5 concentrations over the Beijing–Tianjin–Hebei region in 2013–2018', Atmospheric Chemistry and Physics, 20(13), pp. 7667–7682.
        """)
        st.markdown("---")  # Adds a visual separator between sections
    
    st.markdown("## 🌤️ Seasonal Variation of PM2.5 Across Area Types")
    
    seasonal_variation_image = "seasonal_pm25_by_area.png"
    if os.path.exists(seasonal_variation_image):
        st.image(Image.open(seasonal_variation_image), caption="Seasonal Variation of PM2.5 Levels by Area Type", width=650)
    else:
        st.warning("⚠️ Seasonal variation chart not found.")
    
    with st.expander("👉 Click to see detailed analysis of seasonal pollution patterns"):
        st.markdown("""
        This visualization reveals the complex interplay between seasonal meteorology and geographic location, demonstrating how both temporal and spatial factors combine to create variable pollution exposure patterns across Beijing's diverse areas.

        ### Seasonal Patterns

        The most significant pattern in this data is the pronounced seasonality of PM2.5 concentrations across all area types:

        1. **Winter Pollution Crisis**: Winter shows dramatically elevated PM2.5 levels across all regions, with urban areas reaching a concerning average of ~102 μg/m³. This aligns with research by (Wang et al., 2019), who noted that "winter PM2.5 concentrations in Beijing frequently exceed 100 μg/m³ due to increased coal combustion for heating and unfavorable meteorological conditions including temperature inversions that trap pollutants near the surface"

        2. **Summer Minimum**: Summer shows the lowest PM2.5 concentrations (55-70 μg/m³) across all area types. This seasonal improvement can be attributed to what (Liu et al., 2018) described as "enhanced vertical mixing, increased precipitation scavenging, and reduced emissions from heating sources during summer months in North China"

        3. **Transitional Seasons**: Spring and autumn show intermediate levels, with autumn generally higher than spring across all areas.

        ### Geographic Disparities Persist Across Seasons

        A critical observation is that the relative ranking of area types remains mostly consistent across seasons:

        1. **Urban Vulnerability**: Urban areas consistently show among the highest PM2.5 levels in all seasons, with the urban-rural gap widening most dramatically in winter (~20 μg/m³ difference).

        2. **Industrial Areas**: Industrial zones show high PM2.5 in winter (95 μg/m³) and remain elevated even in summer (65 μg/m³). This pattern supports findings by (Zheng et al., 2017) who reported that "industrial emissions contribute significantly to Beijing's PM2.5 burden throughout the year, but their proportional impact is greater during summer when heating-related emissions decrease"

        3. **Suburban Advantage**: Suburban areas consistently maintain lower PM2.5 levels than both urban and industrial zones across all seasons, suggesting structural advantages in these intermediate locations.

        ### Geographic Resilience to Seasonal Effects

        The data reveals interesting differences in seasonal vulnerability:

        1. **Highest Seasonal Amplitude**: Urban areas show the largest seasonal swing (~40 μg/m³ difference between summer and winter), supporting (Xing, Mao and Duan, 2022) who found that "urban centers in Beijing experience more extreme seasonal variations in PM2.5 compared to surrounding areas due to intensified winter heating emissions and canyon effects that trap pollutants"

        2. **Rural Stability**: Rural areas show the most moderate seasonal changes, maintaining relatively consistent (though still unhealthy) PM2.5 levels year-round.

        ### Health and Policy Implications

        These findings have significant implications for seasonal environmental justice. As noted by (Lei et al., 2022), "the combination of geographic and seasonal disparities in air pollution exposure creates compounded vulnerability for urban residents during winter months, with average exposure exceeding WHO guidelines by more than 20-fold"

        ---

        **References:**
        - Lei, R. et al. (2022) 'Spatial and temporal characteristics of air pollutants and their health effects in China during 2019–2020', Journal of Environmental Management, 317, p. 115460.
        - Liu, H. et al. (2018) 'Ground-level ozone pollution and its health impacts in China', Atmospheric Environment, 173, pp. 223–230.
        - Wang, Y. et al. (2019) 'Trends in particulate matter and its chemical compositions in China from 2013–2017', Science China Earth Sciences, 62(12), pp. 1857–1871.
        - Xing, L., Mao, X. and Duan, K. (2022) 'Impacts of urban–rural disparities in the trends of PM2.5 and ozone levels in China during 2013–2019', Atmospheric Pollution Research, 13(11), p. 101590.
        - Zheng, Y. et al. (2017) 'Air quality improvements and health benefits from China's clean air action since 2013', Environmental Research Letters, 12(11), p. 114020.
        """)
        st.markdown("---")  # Adds a visual separator between sections
    
    st.markdown("## 📈 Monthly Trends of Key Pollutants")
    
    monthly_pollutants_image = "monthly_pollutants_trend.png"
    if os.path.exists(monthly_pollutants_image):
        st.image(Image.open(monthly_pollutants_image), caption="Pollution Levels by Month (Median)", width=650)
    else:
        st.warning("⚠️ Monthly pollutant trends chart not found.")
    
    with st.expander("👉 Click to see detailed analysis of monthly pollutant patterns"):
        st.markdown("""
        The plot displays the monthly median levels of four significant pollutants: SO2, NO2, CO, and O3, providing a comprehensive overview of their seasonal patterns.

        ### Key Observations:

        - **SO2:** Elevated concentrations are observed in the winter months, notably in **January, February, and December**, suggesting increased emissions from heating and lower atmospheric dispersion (Wang et al., 2014).
        
        - **NO2:** Peaks during **January, February, and the late fall months (October to December)**, which may be attributed to heightened vehicular emissions and domestic heating during the colder months (Zheng et al., 2018).
        
        - **CO:** Similarly, CO levels are higher in **January, February, and December**, indicating a correlation with fossil fuel burning for heating and potential traffic emissions (Li et al., 2017).
        
        - **O3:** In contrast to SO2, NO2, and CO, O3 levels reach their maximum in the warmer months, particularly **April to July**, due to increased photochemical reactions under higher sunlight (Li et al., 2017).

        ### Opposing Seasonal Patterns

        The most striking finding is the inverse relationship between O3 and the other pollutants. While SO2, NO2, and CO peak in winter, O3 reaches maximum levels in summer. This inverse pattern has been well-documented in urban environments and reflects fundamental atmospheric chemistry:

        1. **Winter highs for primary pollutants**: SO2, NO2, and CO are directly emitted from combustion sources. Their winter peaks reflect both increased emissions (heating) and reduced atmospheric dispersion due to temperature inversions.
        
        2. **Summer highs for secondary pollutants**: O3 is formed through photochemical reactions requiring sunlight, NOx, and volatile organic compounds. Higher summer temperatures and stronger solar radiation drive these reactions.

        ### Implications for Health and Policy

        These distinct seasonal patterns suggest that Beijing faces different air quality challenges throughout the year:
        
        - **Winter strategy**: Focus on reducing primary emissions from heating and improving ventilation in the urban environment
        - **Summer strategy**: Target precursors of photochemical smog (NOx and VOCs) to mitigate ozone formation
        
        As noted in the literature, effective year-round air quality management requires season-specific approaches rather than one-size-fits-all policies.

        ---

        **References:**
        - Wang, Y. et al. (2014) 'Spatial and temporal variations of six criteria air pollutants in 31 provincial capital cities in China during 2013–2014', Environment International, 73, pp. 413–422.
        - Zheng, B. et al. (2018) 'Trends in China's anthropogenic emissions since 2010 as the consequence of clean air actions', Atmospheric Chemistry and Physics, 18(19), pp. 14095–14111.
        - Li, K. et al. (2017) 'Meteorological and chemical impacts on ozone formation: A case study in Hangzhou, China', Atmospheric Research, 196, pp. 40–52.
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