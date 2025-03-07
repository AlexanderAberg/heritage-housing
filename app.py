import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_machine_learning import page_machine_learning_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_house_prices import page_house_prices_body
from app_pages.page_correlation import page_correlation_body

app = MultiPage(app_name= "HousePrices") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Machine Learning HousePrices", page_machine_learning_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("House Prices", page_house_prices_body)
app.add_page("Correlation Analysis", page_correlation_body)

app.run() # Run the  app