import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from fuctions import *

st.set_page_config(page_title="Auto Data Predict",page_icon="🔮",layout="wide")
st.subheader("⚡️  Analytics Dashboard")
st.markdown("##")

st.sidebar.image("resources/logo.png",caption="Auto Data Predict")

def Home():
    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=0)

    with st.sidebar:
        choice = st.radio('navigation', ['Upload', 'Profiling','Modelling'])

    if choice == 'Upload':
        st.title('Upload Your Dataset (file_name.csv)')
        file = st.file_uploader('Upload DS')
        if file:
            df = pd.read_csv(file, index_col=0)
            df.to_csv('dataset.csv')
            st.dataframe(df)

    if choice == 'Profiling':
        st.title("Exploratory Data Analysis")
        if st.button('Run Profiling'):
            profile_df = ProfileReport(df)
            st_profile_report(profile_df)

    if choice == 'Preprocessing':
        st.title('Data Preprocessing')
        if st.button('Run Preprocessing'):
            df = preprocessing_data(df)
            st.dataframe(df)

            with st.spinner('Loading... ⏳🤖'):
                st.title('OK')

    if choice == 'Modelling':
        st.title('Choose the supervised ML')
        chosen_model = st.radio('', ['Regresion', 'Clasification'])

        if chosen_model == 'Clasification':
            df = preprocessing_data(df)
            st.dataframe(df)
            st.title('Choose the Target Column')
            chosen_target = st.selectbox('', df.columns)
            if st.button('Run Modelling'):
                train_set, val_set, test_set = train_val_test_split(df)
                X_train, y_train, X_val, y_val, X_test, y_test = remove_labels(train_set, val_set, test_set, chosen_target)
                st.dataframe(X_train)
                st.dataframe(y_train)
                df_classifiers = search_model(X_train, y_train, X_val, y_val, X_test, y_test, 1)
                st.dataframe(df_classifiers)

            with st.spinner('Loading... ⏳🤖'):
                st.title('OK')


if __name__ == "__main__":
    Home()
