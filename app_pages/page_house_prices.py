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
    """
    Streamlit page function for predicting the houses sales price.
    """
    st.write("## Sales Price Prediction")

    # Load the pre-trained model pipeline for price prediction
    regression_pipe = load_model()

    # The title for the page
    st.write("## House Price Prediction")

    regression_pipe = load_model()

    X_inherited = load_inherited_houses_data()

    # List of important features for the house price prediction
    house_features = ["1stFlrSF",
                      "BsmtFinSF1", "GarageArea", "GarageYrBlt",
                      "GrLivArea", "LotArea", "OverallQual",
                      "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]


      # Filter the inherited houses dataset to only include important features
    X_inherited_filtered = X_inherited[house_features]

    # Display the dataset with the important features
    st.write("#### Inherited Houses (Filtered Data for Prediction)")
    st.write(X_inherited_filtered)


    predicted_prices = predict_price(X_inherited_filtered, house_features,
                                     regression_pipe)
    
    X_inherited['PredictedSalePrice'] = predicted_prices


    # Display the inherited houses data with the predicted sale prices
    st.write("#### Predicted Sale Prices for the Inherited Houses")
    st.dataframe([["1stFlrSF",
                    "BsmtFinSF1", "GarageArea", "GarageYrBlt",
                    "GrLivArea", "LotArea", "OverallQual",
                    "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]])


    total_price = X_inherited['PredictedSalePrice'].sum()

    st.write(f"### Total Predicted Sale Price for the Inherited Houses: "
             f"**💲{round(total_price, 2):,}**")

    st.write("#### Predict Sales Price for Your Own House")

    X_live = DrawInputsWidgets(house_features)

    # Predict the price for the custom user-provided house
    price_prediction_live = predict_price(X_live, house_features,
                                          regression_pipe)


def DrawInputsWidgets(house_features):
    
    # Load the housing dataset to help determine the realistic input ranges
    df = load_housing_data()

    # Define range for scaling input values
    percentageMin, percentageMax = 0.3, 2.2

    # Reload the pre-trained regression model for predicting the house price
    price_pipeline = joblib.load(
        "outputs/ml_pipeline/predict_price/v1/regression_pipeline.pkl"
    )

    # Initialize an empty DataFrame to store the custom user's house input
    X_live = pd.DataFrame([], index=[0])

    # Set up 10 columns for the user to input the house features 
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

    # Ensure the relevant features are in the correct order for input
    relevant_features = ["1stFlrSF",
                         "BsmtFinSF1", "GarageArea", "GarageYrBlt",
                         "GrLivArea", "LotArea", "OverallQual",
                         "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]

    # Create input widgets for each feature one by one, allowing the user to
    # set their house details
    with col1:
        feature = relevant_features[0]
        st_widget = st.number_input(
            label=feature,
             min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=10
        )
        X_live[feature] = st_widget

    with col2:
        feature = relevant_features[1]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=10
        )
        X_live[feature] = st_widget

    with col3:
        feature = relevant_features[2]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=10
        )
        X_live[feature] = st_widget

    with col4:
        feature = relevant_features[3]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min() * percentageMin),
            max_value=date.today().year,
            value=int(df[feature].median()),
            step=1
        )
        X_live[feature] = st_widget

    with col5:
        feature = relevant_features[4]
        st_widget = st.number_input(
            label=feature,
             min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=10
        )
        X_live[feature] = st_widget

    with col6:
        feature = relevant_features[5]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=100
        )
        X_live[feature] = st_widget

    with col7:
        feature = relevant_features[6]
        st_widget = st.number_input(
            label=feature,
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
        X_live[feature] = st_widget

    with col8:
        feature = relevant_features[9]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min() * percentageMin),
            max_value=date.today().year,
            value=int(df[feature].median()),
            step=1
        )
        X_live[feature] = st_widget

    with col9:
        feature = relevant_features[8]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min() * percentageMin),
            max_value=date.today().year,
            value=int(df[feature].median()),
            step=1
        )
        X_live[feature] = st_widget

    with col10:
        feature = relevant_features[9]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min() * percentageMin),
            max_value=date.today().year,
            value=int(df[feature].median()),
            step=1
        )
        X_live[feature] = st_widget

    # Button to calculate the predicted price based on inputs and display
    if st.button('Calculate the House Price'):
        predicted_price = (X_live, house_features, price_pipeline)

        # Display the predicted house price on the Streamlit page
        st.write(f"### Calculated Successfully \n"
                 f"### The price for your house:"
                 f"💲{round(predicted_price[0], 2):,}")

    return X_live