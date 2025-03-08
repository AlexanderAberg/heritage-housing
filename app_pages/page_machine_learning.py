import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src.data_management import load_ames_data, load_pkl_file
from src.machine_learning.evaluate_reg import regression_performance

def load_pkl_file(file_path):
    """
    Loads a pickled file (model) using joblib from the given file path.

    """
    try:
        # Attempt to load the pickled file using joblib
        model = joblib.load(file_path)
        return model
    except Exception as e:
        # If an error occurs during loading, print the error and return None
        print(f"Error loading the model from {file_path}: {e}")
        return None


def page_machine_learning_body():
    """
    Is displaying the machine learning pipeline and regression
    performance plots.
    """
    # Define the version of the pipeline being used
    version = 'v1'

    # Load the regression pipeline (model)
    price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl"
    )

    price_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_price/{version}/features_importance.png"
    )

    # Load the training and test datasets (features and targets)
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/X_train.csv"
    )
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/X_test.csv"
    )
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/y_train.csv"
    )
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/y_test.csv"
    )

    # Display the title for the ML pipeline section
    st.write("## ML Pipeline: Predict House Price")

    # Show an introductory message about the ML pipeline
    st.info(
        "* To answer the Business Requirement 2, a Regressor model was trained" 
        " and the pipeline tuned aiming for at least 0.9 accuracy in "
        "predicting the sales price of a property with a set of attributes.\n"
        "* The pipeline performance for the best model on the train and test "
        "set is R2 = 1.0 (flawless) and R2 =0.883 (almost 0.9) respectively.\n"
        "* We present the pipeline steps, best features list along with "
        "feature importance plot, pipeline performance and regression "
        "performance report below."
    )
    st.write("---")

    # Display the ML pipeline code
    st.write("* ML pipeline to predict sales prices of houses ")
    st.code(price_pipe)  # Display the code of the pipeline
    st.write("---")

    st.write("* The features the model was trained and their importance")
    st.write(X_train.columns.to_list())  # The features used for training
    st.write("---")

    # Information about performance goals for the pipeline
    st.write("### Pipeline Performance")
    st.write("##### Performance goal of the predictions:")
    st.write(
        "* We agreed with the client an R2 average score of at least 0.9 on "
        "the train set and the test set."
    )
    st.write(
        "* Our Machine Learning pipeline performance shows that our "
        "model performance metrics have achieved the requirements, "
        "thanks to the flawless score on the train model."
    )

    # The regression performance plots
    st.write("### Regression Performance Plots")
    st.write(
        "* The regression performance plots below indicate that the model "
        "with the best features can predict sale prices well. For houses "
        "with higher prices, the model does, however, look to be less "
        "dependable."
    )

    # The regression performance plot
    image_path = 'docs/plots/regression_performance.png'
    st.image(image_path, caption="Regression Performance",
             use_container_width=True)