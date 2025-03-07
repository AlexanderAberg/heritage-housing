import streamlit as st
import numpy as np
import pandas as pd
import pickle as px
import matplotlib.pyplot as plt
import seaborn as sns

model_path = f"outputs/ml_pipeline//predict_price/v1/regression_pipeline.pkl"

def page_cluster_body():

    

    # dataframe for cluster_distribution_per_variable()

    st.write("### ML Pipeline: Cluster Analysis")
    # display pipeline training summary conclusions
    st.info(
        f"* We refitted the cluster pipeline using fewer variables, and it delivered equivalent "
        f"performance to the pipeline fitted using all variables.\n"
        f"* The pipeline average silhouette score "
    )
    st.write("---")


    # text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    st.write("#### Cluster Profile")
    statement = (
      
    )
    st.info(statement)

    # text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    statement = (
        f"* The cluster profile interpretation allowed us to label the cluster in the following fashion:\n"
    
    )
    st.success(statement)

    # hack to not display the index in st.table() or st.write()


# code coped from "07 - Modeling and Evaluation - Cluster Sklearn" notebook - under "Cluster Analysis" section
def cluster_distribution_per_variable(df, target):

    df_bar_plot = df.groupby(["SalePrice", target]).size().reset_index(name="Count")
    df_bar_plot.columns = ['SalePrice', target, 'Count']
    df_bar_plot[target] = df_bar_plot[target].astype('object')

    st.write(f"#### SalePrice distribution across {target} levels")
    fig = px.bar(df_bar_plot, x='SalePrice', y='Count',
                 color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['SalePrice'].unique()))
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)

    df_relative = (df
                   .groupby(["SalePrice", target])
                   .size()
                   .unstack(fill_value=0)
                   .apply(lambda x:  100*x / x.sum(), axis=1)
                   .stack()
                   .reset_index(name='Relative Percentage (%)')
                   .sort_values(by=['SalePrice', target])
                   )
    df_relative.columns = ['SalePrice', target, 'Relative Percentage (%)']

    st.write(f"#### Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x='SalePrice', y='Relative Percentage (%)',
                  color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['SalePrice'].unique()))
    fig.update_traces(mode='markers+lines')
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)