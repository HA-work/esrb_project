import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json

st.set_page_config(
    page_title="Original Data"
    )


st.header("Original Video Game Data")

st.write("This is the original data.")

original_data = "https://raw.githubusercontent.com/HA-work/esrb_project/main/data/video_games_esrb_rating.csv"


df = pd.read_csv(original_data)
st.dataframe(data=df)