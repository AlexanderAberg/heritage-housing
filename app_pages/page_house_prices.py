import joblib
from datetime import date
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_management import load_housing_data, load_inherited_houses_data
from src.machine_learning.predictive_analysis import predict_price

def load_model():
    """
    """
    file_path = "outputs/ml_pipeline/predict_price/v1"
    return joblib.load(f"{file_path}/regression_pipeline.pkl")


def page_house_prices_body():
    st.write("## Sales Price Prediction")
    
    # Load model once
    regression_pipe = load_model()
    
    # Get inherited houses data
    X_inherited = load_inherited_houses_data()

    # List of important features for house price prediction
    house_features = ["1stFlrSF", "BsmtFinSF1", "GarageArea", "GarageYrBlt",
                     "GrLivArea", "LotArea", "OverallQual",
                     "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]
    
    # Filter the inherited houses dataset to only include important features
    X_inherited_filtered = X_inherited[house_features]

    # Display the dataset with the important features
    st.write("#### Inherited Houses (Filtered Data for Prediction)")
    st.write(X_inherited_filtered)
    
     # Calculate predictions for inherited houses
    predicted_prices = predict_price(X_inherited_filtered, house_features, regression_pipe)
    X_inherited['PredictedSalePrice'] = predicted_prices
    
    # Display the inherited houses data with predicted prices
    total_price = X_inherited['PredictedSalePrice'].sum()
    st.write(f"### Total Predicted Sale Price for the Inherited Houses:")
    st.write(f"**💲{round(total_price, 2):,}**")
    
    # Get customers house inputs
    st.write("#### Predict Sales Price for Your Own House")
    X_live = DrawInputsWidgets(house_features)
    
    # Predict price for customers house
    if st.button('Calculate the House Price'):
        try:
            predicted_price = predict_price(X_live, house_features, regression_pipe)[0]
            st.write(f"### Calculated Successfully \\n"
                    f"### The price for your house:")
            st.write(f"💲{round(predicted_price, 2):,}")
        except Exception as e:
            st.error(f"Error calculating prediction: {str(e)}")

def DrawInputsWidgets(house_features):
    df = load_housing_data()
    percentageMin, percentageMax = 0.3, 2.2
    
    X_live = pd.DataFrame([], index=[0])

    # Create columns one time
    cols = st.columns(10)
    
    # Dictionary mapping features to their specific parameters
    feature_params = {
        "OverallQual": {"min_val": 1, "max_val": 10, "step": 1},
        "YearBuilt": {"min_val": int(df["YearBuilt"].min() * percentageMin),
                     "max_val": date.today().year,
                     "step": 1},
        "YearRemodAdd": {"min_val": int(df["YearRemodAdd"].min() * percentageMin),
                        "max_val": date.today().year,
                        "step": 1}
    }
    
    # Create widgets that has unique keys
    for idx, feature in enumerate(house_features):
        col = cols[idx % len(cols)]
        params = feature_params.get(feature,
                                  {"min_val": int(df[feature].min()),
                                   "max_val": int(df[feature].max()),
                                   "step": 1})
        
        with col:
            st_widget = st.number_input(
                label=feature,
                min_value=params["min_val"],
                max_value=params["max_val"],
                value=int(df[feature].median()),
                step=params["step"],
                key=f"{feature}_input"
            )
            X_live[feature] = st_widget
            
    return X_live