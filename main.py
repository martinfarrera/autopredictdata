import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from fuctions import *

st.set_page_config(page_title="Auto Data Predict",page_icon="⚡️",layout="wide")
st.subheader("⚡️ Analytics Dashboard")
st.markdown("##")

st.sidebar.image("resources/logo.png")

def upload():
    os.path.exists('./dataset.csv')
    df = pd.read_csv('dataset.csv', index_col=0)
    st.dataframe(df)


def sideBar():

 with st.sidebar:
    selected=option_menu(
        menu_title="Menu",
        options=["Upload","Profiling"],
        icons=["house","eye"],
        menu_icon="cast",
        default_index=0
    )
 if selected=="Upload":
     upload()
 if selected=="Profiling":
    Progressbar()
    graphs()


if __name__ == "__main__":
    sideBar()