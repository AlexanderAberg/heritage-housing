import streamlit as st

def page_summary_body():
    st.header("Project Summary")
    st.info("""
             **General info:**
            The page goal is to be able to estimate the housing prices in Aimes based on different attributes including for example something obvious like size but also less obvious 
            such as when the garage was built.
            """)
    
    st.success("""
                **Dataset:**
                The dataset is taken from https://www.kaggle.com/datasets/codeinstitute/housing-prices-data
               
                [Licence](https://github.com/)
               """)
    
    st.warning("""
                **Business requirements:**
                - The client requires analysis and visualization of the dataset.
                - The client expects a machine learning model with an accuracy above 80%.
               """)
    # Link to README file
    st.markdown("Read the full README [here](https://github.com/)", unsafe_allow_html=True)
    
    
    

    