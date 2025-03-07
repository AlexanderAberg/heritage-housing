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
                "***Business Requirement 1**: Explore and analyze how various "
                "house characteristics impact the sale price, with the help of "
                "visualizations to illustrate these connections.\n"
                "***Business Requirement 2**: Build a forecasting model to predict "
                "the sale prices of the inherited properties and other homes in Ames, "
                "Iowa."
                """)
    # Link to README file
    st.markdown("Read the full README [here](https://github.com/AlexanderAberg/heritage-housing/blob/main/README.md)", unsafe_allow_html=True)
    
    
    

    