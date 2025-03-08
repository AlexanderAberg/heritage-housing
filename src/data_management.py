import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Caching ensures data is not reloaded unnecessarily in Streamlit
@st.cache_data
def load_housing_data():
    """
    Load the housing dataset from a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the housing dataset in Ames, Iowa.
    """
    return pd.read_csv("outputs/datasets/cleaned/HousePricesCleaned.csv")


@st.cache_data
def load_inherited_houses_data():
    """
    Load the inherited houses dataset from a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the inherited houses dataset.
    """
    return pd.read_csv("outputs/datasets/collection/InheritedHouses.csv")


def load_pkl_file(file_path):
    """
    Load a serialized object from a pickle (.pkl) file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        object: The deserialized object from the pickle file.
    """
    return joblib.load(filename=file_path)