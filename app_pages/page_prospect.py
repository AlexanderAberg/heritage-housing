import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
# import joblib

model_path = f"outputs/ml_pipeline/HousePrices/v1/clf_pipeline_data_cleaning_feat_eng.pkl"
model_path = f"outputs/ml_pipeline/cluster_analysis/v1/cluster_pipeline.pkl"
model_path = f"outputs/ml_pipeline/predict_tenure/v1/label_map.pkl"

def page_prospect_body():

    # load predict SalePrice files
    # version = 'v1'
    # SalePrice_dc_fe = joblib.load(
    #     f"outputs/ml_pipeline/HousePrices/v1/clf_pipeline_data_cleaning_feat_eng.pkl")
    # SalePrice_pipe_model = joblib.load(
    #     f"outputs/ml_pipeline/HousePrices/v1/clf_pipeline_data_cleaning_feat_eng.pkl")
    # SalePrice_features = (pd.read_csv(f"outputs/ml_pipeline/HousePrices/v1/X_train.csv")
    #                   .columns
    #                   .to_list()
    #                   )

    # load predict tenure files
    # version = 'v1'
    # SalePrice_pipe_dc_fe = joblib.load(
    #     f"outputs/ml_pipeline/predict_tenure/v1/label_map.pkl")
    # tenure_labels_map = joblib.load(
    #      f"outputs/ml_pipeline/predict_tenure/v1/label_map.pkl")
    # tenure_features = (pd.read_csv(f"outputs/ml_pipeline/predict_tenure/v1/X_train.csv")
    #                    .columns
    #                    .to_list()
    #                    )

    # load cluster analysis files
    # version = 'v1'
    # cluster_pipe = joblib.load(
    #     f"outputs/ml_pipeline/cluster_analysis/v1/cluster_pipeline.pkl")
    # cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/v1/cluster_pipeline.pkl")
    #                     .columns
    #                     .to_list()
    #                     )
    # cluster_profile = pd.read_csv(
    #     f"outputs/ml_pipeline/cluster_analysis/v1/cluster_pipeline.pkl")

    st.write("### Prospect Churnometer Interface")
    st.info(
        f"* The client is interested in determining whether or not a given prospect will churn. "
        f"If so, the client is interested to know when. In addition, the client is "
        f"interested in learning from which cluster this prospect will belong in the customer base. "
        f"Based on that, present potential factors that could maintain and/or bring  "
        f"the prospect to a non-churnable cluster."
    )
    st.write("---")

    # Generate Live Data
    # check_variables_for_UI(tenure_features, churn_features, cluster_features)
    X_live = DrawInputsWidgets()

    # predict on live data
    # if st.button("Run Predictive Analysis"):
    #     SalePrice_prediction = predict_SalePrice(
    #         X_live, SalePrice_features, SalePrice_pipe_dc_fe, SalePrice_pipe_model)

    #     if churn_prediction == 1:
    #         predict_tenure(X_live, tenure_features,
    #                        tenure_pipe, tenure_labels_map)

    #     predict_cluster(X_live, cluster_features,
    #                     cluster_pipe, cluster_profile)


def check_variables_for_UI(tenure_features, SalePrice_features, cluster_features):
    import itertools

    # The widgets inputs are the features used in all pipelines (tenure, churn, cluster)
    # We combine them only with unique values
    combined_features = set(
        list(
            itertools.chain(tenure_features, SalePrice_features, cluster_features)
        )
    )
    st.write(
        f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")


def DrawInputsWidgets():

    # load dataset
    percentageMin, percentageMax = 0.4, 2.0

# we create input widgets only for 6 features
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)

    # We are using these features to feed the ML pipeline - values copied from check_variables_for_UI() result

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type (numerical or categorical)
    # and set initial values


    # st.write(X_live)

    return X_live