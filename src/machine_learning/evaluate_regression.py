import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            r2_score, explained_variance_score, \
                            median_absolute_error


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    """
    Evaluates the performance of a regression model on the training dataset and the test
    dataset.
    """
    st.write("Model Evaluation \n")
    st.info("* Train Set")
    regression_evaluation(X_train, y_train, pipeline)
    st.info("* Test Set")
    regression_evaluation(X_test, y_test, pipeline)


def regression_evaluation(X, y, pipeline):
    """
    Computes and displays various regression evaluation metrics for a dataset.
    """
    # Generate predictions using the pipeline model
    prediction = pipeline.predict(X)

    # If target variable is a DataFrame, extract the values
    if isinstance(y, pd.DataFrame):
        y = y['SalePrice'].values

    # Compute evaluation metrics
    medae = median_absolute_error(y, prediction) 
    mape = np.mean(np.abs((y - prediction) / y)) * 100
    mae = mean_absolute_error(y, prediction) 
    mse = mean_squared_error(y, prediction) 
    rmse = np.sqrt(mse) 
    r2 = r2_score(y, prediction) 
    evs = explained_variance_score(y, prediction) 

    # Displaying the metrics
    st.write('Median Absolute Error:', round(medae, 3))
    st.write('Mean Absolute Percentage Error:', round(mape, 3))
    st.write('Mean Absolute Error:', round(mae, 3))
    st.write('Mean Squared Error:', round(mse, 3))
    st.write('Root Mean Squared Error:', round(rmse, 3))
    st.write('R2 Score:', round(r2, 3))
    st.write('Explained Variance Score:', round(evs, 3))
    st.write("\n")