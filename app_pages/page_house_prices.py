import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

model_path = f"outputs/ml_pipeline/HousePrices/v1/clf_pipeline_data_cleaning_feat_eng.pkl"



def page_house_prices_body():

    # load needed files
    X_train = pd.read_csv(
        "outputs/ml_pipeline/HousePrices/v1/X_train.csv")
    X_test = pd.read_csv(
        "outputs/ml_pipeline/HousePrices/v1/X_test.csv")
    y_train = pd.read_csv(
        "outputs/ml_pipeline/HousePrices/v1/y_train.csv").values
    y_test = pd.read_csv(
        "outputs/ml_pipeline/HousePrices/v1/y_test.csv").values

    st.write("### ML Pipeline: Predict SalePrice")
    # display pipeline training summary conclusions
    st.info(
        f"* The pipeline was tuned aiming at least 0.80 predicting SalePrice "
    )

    # show pipelines
    st.write("---")
    st.write("#### There are 2 ML Pipelines arranged in series.")

    st.write(" * The first is responsible for data cleaning and feature engineering.")

    st.write("* The second is for feature scaling and modelling.")

    # show feature importance plot
    st.write("---")
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())

    # We don't need to apply dc_fe pipeline, since X_train and X_test
    # were already transformed in the jupyter notebook (Predict Customer Churn.ipynb)

    # evaluate performance on train and test set
    st.write("---")
    st.write("### Pipeline Performance")
    # clf_performance(X_train=X_train, y_train=y_train,
    #                 X_test=X_test, y_test=y_test,
    #                 pipeline=model_path,
    #                 label_map=["SalePrice"])