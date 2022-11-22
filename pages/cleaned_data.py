import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json


st.set_page_config(
    page_title="Cleaned Data"
    )


st.header("Cleaned Video Game Data")

#

st.write("This is the data after removing duplicates, fixing spelling errors and correcting wrong data values.")

st.write("Additional features were also added like the Number of Descriptors")


cleaned_data = "https://raw.githubusercontent.com/HA-work/esrb_project/main/data/cleaned_games.csv"


df = pd.read_csv(cleaned_data)
st.dataframe(data=df)