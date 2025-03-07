import joblib
from datetime import date
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    """
    Loads and returns the pre-trained regression model pipeline from a pickle
    file.The pipeline is used for predicting house prices based on input
    features.
    """
    file_path = "outputs/ml_pipeline/predict_price/v1"
    return joblib.load(f"{file_path}/regression_pipeline.pkl")


def page_house_prices_body():
    """
    Streamlit page function for predicting the sales price of houses.
    This includes predictions for both inherited houses (from dataset)
    and a custom house input by the user.
    """
    # Title for the sales price prediction page
    st.write("## Sales Price Prediction")

    # Load the pre-trained model pipeline for price prediction
    regression_pipe = load_model()


    # List of important house features for price prediction
    house_features = ["1stFlrSF",
                      "BsmtFinSF1", "GarageArea", "GarageYrBlt",
                      "GrLivArea", "LotArea", "OverallQual"
                      "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]


    # Predict the sale price for each inherited house using the model and
    # filtered features
    predicted_prices = ( house_features,
                                     regression_pipe)


    # Display the inherited houses data with the predicted sale prices
    st.write("#### Predicted Sale Prices for Inherited Houses")
    st.dataframe([["1stFlrSF",
                    "BsmtFinSF1", "GarageArea", "GarageYrBlt",
                    "GrLivArea", "LotArea", "OverallQual"
                    "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]])



    # Display the total predicted sale price for all inherited houses
    st.write(f"### Total Predicted Sale Price for All Inherited Houses: "
             f"**💲{round(2):,}**")

    # Title for the random house prediction section (for user input)
    st.write("#### Predict Sales Price for Your Own House")

    # Predict the price for the custom user-provided house
    price_prediction_live = (house_features,
                                          regression_pipe)


def DrawInputsWidgets(house_features):
    """
    Creates interactive input widgets in the Streamlit UI for users to input
    their house features and predict the sales price for their own house.
    """
    # Load the Ames dataset to help determine realistic input ranges
    df = ()
    # Define range for scaling input values
    percentageMin, percentageMax = 0.4, 2.0

    # Reload the pre-trained regression model for predicting the house price
    price_pipeline = joblib.load(
        "outputs/ml_pipeline/predict_price/v1/regression_pipeline.pkl"
    )

    # Initialize an empty DataFrame to store the custom user's house input
    X_live = pd.DataFrame([], index=[0])

    # Set up 5 columns for the user to input their house features interactively
    col1, col2, col3, col4, col5 = st.columns(5)

    # Ensure the relevant features are in the correct order for input
    relevant_features = ["1stFlrSF",
                         "BsmtFinSF1", "GarageArea", "GarageYrBlt",
                         "GrLivArea", "LotArea", "OverallQual"
                         "TotalBsmtSF", "YearBuilt", "YearRemodAdd"]

    # Create input widgets for each feature one by one, allowing the user to
    # set their house details
    with col1:
        feature = relevant_features[0]
        st_widget = st.number_input(
            label=feature,
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
        X_live[feature] = st_widget

    with col2:
        feature = relevant_features[1]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min() * percentageMin),
            max_value=int(df[feature].max() * percentageMax),
            value=int(df[feature].median()),
            step=50
        )
        X_live[feature] = st_widget

    with col3:
        feature = relevant_features[2]
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median()),
            step=50
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
            min_value=int(df[feature].min() * percentageMin),
            max_value=int(df[feature].max() * percentageMax),
            value=int(df[feature].median()),
            step=50
        )
        X_live[feature] = st_widget

    # Button to calculate and display the predicted price based on inputs
    if st.button('Calculate the House Price'):
        # Use the predict price function to calculate the predicted price for
        # the custom house
        predicted_price = (X_live, house_features, price_pipeline)

        # Display the predicted house price on the Streamlit page
        st.write(f"### Calculated Successfully \n"
                 f"### The price for your house:"
                 f"💲{round(predicted_price[0], 2):,}")

    return X_live