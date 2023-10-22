import os
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

def main():
    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=0)

    with st.sidebar:
        choice = st.radio('navigation', ['Upload', 'Profiling'])

    if choice == 'Upload':
        st.title('Upload Your Dataset (file_name.csv)')
        file = st.file_uploader('Upload DS')
        if file:
            df = pd.read_csv(file, index_col=0)
            df.to_csv('dataset.csv')
            st.dataframe(df)

    if choice == 'Profiling':
        st.title("Exploratory Data Analysis")
        profile_df = ProfileReport(df)
        st_profile_report(profile_df)

if __name__ == "__main__":
    main()
