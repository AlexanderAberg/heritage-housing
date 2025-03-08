import streamlit as st


def predict_price(X_live, house_features, price_pipeline):
    """
    Predicts the price of a house based on live data input.

    Parameters:
    X_live (DataFrame): The live input data containing the house features.
    house_features (list): List of the relevant features for the prediction of the price.
    price_pipeline (Pipeline): Pre-trained machine learning pipeline for price
    prediction.

    Returns:
    ndarray: Predicted price(s) for the input house data.
    """
    # Select relevant features from the live input data
    X_live_price = X_live.filter(house_features)

    # Make a prediction using the pre-trained pipeline
    price_prediction = price_pipeline.predict(X_live_price)

    return price_prediction


def predict_inherited_house_price(X_inherited, house_features, price_pipeline):
    """
    Predicting the price of an inherited house based on the features.

    Parameters:
    X_inherited (DataFrame): Data containing features of the inherited house.
    house_features (list): List of relevant features for the price prediction.
    price_pipeline (Pipeline): Pre-trained machine learning pipeline for the price
    prediction.

    Returns:
    float: The predicted price of the inherited house.
    """
    # Select relevant features from the inherited house data
    X_inherited_price = X_inherited.filter(house_features)

    # Make a price prediction using the pre-trained pipeline
    price_prediction_inherited = price_pipeline.predict(X_inherited_price)

    # Extract the predicted price (assuming a single value output)
    the_price = price_prediction_inherited[0]

    return the_price