import streamlit as st
from feature_engine.discretisation import ArbitraryDiscretiser
import numpy as np
import pandas as pd
import plotly.express as px
import ppscore as pps
from src.data_management import load_housing_data

model_path = f"outputs/ml_pipeline/predict_price/v1/regression_pipeline.pkl"

def page_correlation_body():

    df = load_housing_data()

    # List of variables to study in the analysis
    vars_to_study = [
        {'1stFlrSF',
        'BsmtFinSF1', 'GarageArea', 'GarageYrBlt',
        'OpenPorchSF', 'OverallQual', 'TotalBsmtSF',
        'YearBuilt', 'YearRemodAdd'}]

    # Displaying initial text and explanation
    st.write("## Correlation Analysis")
    st.write("")

    st.write("---")

    # Summary of the correlation study and key variables
    st.write("")
    st.write(f"**{vars_to_study}**")

    # Key conclusions based on the correlation study
    st.info(
        ""
    )

    # Subset of data for analysis (target variable and selected features)
    target_var = 'SalePrice'
    st.write("#### Data Visualizations")

    # Display histogram of the target variable's distribution
    if st.checkbox(" Distribution of Target Variable"):
        st.write("")
        plot_target_hist(df, target_var)

    # Display correlation and PPS (Predictive Power Score) heatmaps
    if st.checkbox("Correlation and PPS Heatmap"):
        df_corr_pearson, df_corr_spearman, pps_matrix = (
            CalculateCorrAndPPS(df))
        DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix,
                          CorrThreshold=0.3, PPS_Threshold=0.2)

    # Function to display scatter plots or categorical distribution plots
    def scatter_plot_for_eda(df, col, target_var):
       
        fig = px.scatter(df, x=col, y=target_var,
                         title=f"Scatter Plot of {col} vs {target_var}",
                         trendline="ols", trendline_color_override="orange")
        st.plotly_chart(fig)

    def plot_categorical(df, col, target_var):
        """
        Function to create a stacked histogram for categorical variables vs
        target.
        """
        fig = px.histogram(df, x=col, color=target_var,
                           title=f"Distribution of {col} vs {target_var}",
                           barmode='stack')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

    def variables_plots(df_eda):
    
        target_var = 'SalePrice'
        # Iterate over all variables and plot according to type
        for col in df_eda.drop([target_var], axis=1).columns.to_list():
            if df_eda[col].dtype == 'object':
                plot_categorical(df_eda, col, target_var)
            else:
                scatter_plot_for_eda(df_eda, col, target_var)

    # Display visual analysis for each selected variable
    if st.checkbox(" Visual Analysis"):
        variables_plots(df)


def plot_target_hist(df, target_var):
  
    fig = px.histogram(df, x=target_var, marginal="box", nbins=50,
                       title=f"Distribution of {target_var}")
    st.plotly_chart(fig)


def heatmap_corr(df, threshold, title, figsize=(22, 13), font_annot=9):

    if len(df.columns) > 1:
        # Create a mask to hide upper triangle and correlations
        mask = np.zeros_like(df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        # Apply mask to the correlation matrix
        df_masked = df.mask(mask)

        # Plot heatmap using Plotly
        fig = px.imshow(
            df_masked,
            title=title,
            color_continuous_scale='plasma',
            labels={'x': 'Features', 'y': 'Features'},
            text_auto=True
        )
        st.plotly_chart(fig)


def heatmap_pps(df, threshold, figsize=(22, 13), font_annot=9):
    """
    Generates the Predictive Power Score heatmap.
    """

    if len(df.columns) > 1:
        # Create a mask to hide PPS below the threshold
        mask = np.zeros_like(df, dtype=bool)
        mask[abs(df) < threshold] = True

        # Apply mask to the PPS matrix
        df_masked = df.mask(mask)

        # Plot heatmap using Plotly
        fig = px.imshow(
            df_masked,
            title="Predictive Power Score (PPS) Heatmap",
            color_continuous_scale='plasma',
            labels={'x': 'Features', 'y': 'Features'},
            text_auto=True
        )
        st.plotly_chart(fig)


def CalculateCorrAndPPS(df):
    """
    Function to calculate the Spearman and Pearson correlations and
    the Predictive Power Score matrix for the dataset.
    """
    # Calculate Spearman and Pearson
    df_corr_spearman = df.corr(method="spearman")
    df_corr_spearman.name = 'corr_spearman'
    df_corr_pearson = df.corr(method="pearson")
    df_corr_pearson.name = 'corr_pearson'

    # Calculate Predictive Power Score (PPS)
    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')

    # Display PPS statistics for values below 1
    pps_score_stats = (pps_matrix_raw.query("ppscore < 1").filter(['ppscore'])
                       .describe().T)
    print(pps_score_stats.round(3))

    return df_corr_pearson, df_corr_spearman, pps_matrix


def DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix,
                      CorrThreshold, PPS_Threshold,
                      figsize=(22, 13), font_annot=9):
    """
    Function to display both the Spearman and Pearson correlation heatmap
    and the PPS heatmap.
    """
    # Display Pearson correlation heatmap
    heatmap_corr(df=df_corr_pearson, threshold=CorrThreshold,
                 title="Pearson Correlation Heatmap", figsize=figsize,
                       font_annot=font_annot)

    # Display Spearman correlation heatmap
    heatmap_corr(df=df_corr_spearman, threshold=CorrThreshold,
                 title="Spearman Correlation Heatmap", figsize=figsize,
                 font_annot=font_annot)

    # Display PPS heatmap
    heatmap_pps(df=pps_matrix, threshold=PPS_Threshold, figsize=figsize,
                font_annot=font_annot)