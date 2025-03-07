import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_prospect import page_prospect_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_house_prices import page_house_prices_body
from app_pages.page_cluster import page_cluster_body

app = MultiPage(app_name= "HousePrices") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Prospect HousePrices", page_prospect_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("ML: House Prices", page_house_prices_body)
app.add_page("ML: Cluster Analysis", page_cluster_body)

app.run() # Run the  app