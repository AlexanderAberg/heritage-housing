import streamlit as st
from feature_engine.discretisation import ArbitraryDiscretiser
import numpy as np
import pandas as pd
import plotly.express as px
import ppscore as pps

model_path = f"outputs/ml_pipeline//predict_price/v1/regression_pipeline.pkl"

def page_correlation_body():

    # List of variables to study in the analysis
    vars_to_study = [
        {'1stFlrSF',
        'BsmtFinSF1', 'GarageArea', 'GarageYrBlt',
        'GrLivArea', 'LotArea', 'MasVnrArea',
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

    # Function to display scatter plots or categorical distribution plots
    def scatter_plot_for_eda(col, vars_to_study):
        """
        Function to create a scatter plot between a feature and the target
        variable.
        """
        fig = px.scatter(x=col, y=vars_to_study,
                         title=f"Scatter Plot of {col} vs {vars_to_study}",
                         trendline="ols", trendline_color_override="red")
        st.plotly_chart(fig)

    def plot_categorical(col, vars_to_study):
        """
        Function to create a stacked histogram for categorical variables vs
        target.
        """
        fig = px.histogram(x=col, color=vars_to_study,
                           title=f"Distribution of {col} vs {vars_to_study}",
                           barmode='stack')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

    def variables_plots():
        """
        Function to plot either scatter plots or categorical plots for all
        selected variables.
        """
        vars_to_study = 'SalePrice'
      
    # Display visual analysis for each selected variable
    if st.checkbox(" Variables Plots - Visual Analysis"):
        variables_plots()


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    """
    Function to generate a Predictive Power Score (PPS) heatmap.
    Values below the specified threshold are hidden in the plot.

    Args:
        df: DataFrame containing the PPS matrix.
        threshold: PPS value below which cells will be hidden.
        figsize: Optional tuple to set the size of the heatmap.
        font_annot: Optional font size for annotations in the heatmap.
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
            title="Predictive Power Score Heatmap",
            color_continuous_scale='viridis',
            labels={'x': 'Features', 'y': 'Features'},
            text_auto=True
        )
        st.plotly_chart(fig)


def CalculateCorrAndPPS(df):
    """
    Function to calculate both the Pearson and Spearman correlations as well as
    the Predictive Power Score (PPS) matrix for the dataset.

    Args:
        df: DataFrame containing the features to analyze.

    Returns:
        df_corr_pearson: Pearson correlation matrix.
        df_corr_spearman: Spearman correlation matrix.
        pps_matrix: Matrix of PPS values between the features.
    """
    # Calculate correlation matrices
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
                      figsize=(20, 12), font_annot=8):
    """
    Function to display both the correlation heatmap and the PPS heatmap.

    Args:
        df_corr_pearson: Pearson correlation matrix.
        df_corr_spearman: Spearman correlation matrix.
        pps_matrix: Matrix of PPS values between the features.
        CorrThreshold: Threshold to mask correlations in the heatmap.
        PPS_Threshold: Threshold to mask PPS values in the heatmap.
        figsize: Optional tuple for setting the size of the heatmaps.
        font_annot: Optional font size for annotations in the heatmap.
    """
    # Display PPS heatmap
    heatmap_pps(df=pps_matrix, threshold=PPS_Threshold, figsize=figsize,
                font_annot=font_annot)