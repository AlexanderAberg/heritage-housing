import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    st.success(
        "* We believe that the size of the house and its features affect the "
        "SalePrice "
        "We notice that it does, but it is not the most important feature, "
        "because quality has a higher affection and YearBuilt has "
        "a bigger importance than a few size related features \n\n"

        "* Our conclusion is that you have to take care of your "
        "to get a good SalePrice "
        "This means that a big house with a big garage, porch, yard etc. "
        "will get a very high SalePrice if you are also keeping the house in "
        "great condition \n\n"
        "We think that one of the reason YearBuilt has a big affect because a "
        "new house is often in better condition,"
        "that can also be a reason why a house from 1940 "
        "has about the same SalePrice as one from 1960"
        "The YearRemodAdd also have a decently high correlation, but "
        "much less because a variety of the quality of the house work, while "
        "a new house has to a big degree professionals building the house"
    )
