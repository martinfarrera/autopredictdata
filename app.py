import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from fuctions import *

st.set_page_config(page_title="Auto Data Predict",page_icon="🔮️",layout="wide")
st.title("️️🔮️ Auto Data Predict")
st.markdown("##")

st.sidebar.image("utils/logo.png")
st.sidebar.markdown("##")

def info():
    st.subheader("Descripción")
    st.markdown("Un programa de machine learning predice datos mediante la recopilación de información histórica y variables relevantes. Los datos se procesan y se entrena un modelo con un algoritmo adecuado. Luego, se evalúa su rendimiento, se ajusta si es necesario y se utiliza para hacer predicciones futuras en un entorno de producción.")

def upload():
    st.subheader('Upload Your DataSet (file_name.csv)')
    file = st.file_uploader('Select DS')
    if file:
        df = pd.read_csv(file, index_col=0)
        st.dataframe(df)
        df.to_csv('./DS/ds_upload.csv')

    st.markdown("##")
    st.subheader("Dataset de Prueba")
    st.markdown("Si no cuentas con una usa este DS sobre los acidentes cerebrovasculares.")
    df_test = pd.read_csv('./resources/healthcare-stroke-data.csv', index_col=0)
    st.dataframe(df_test)
    if st.button('Usa este DataSet Test'):
        df_test.to_csv('./DS/ds_upload.csv')

def profiling():
    st.subheader("Exploratory Data Analysis")

    try:
        df = pd.read_csv('./DS/ds_upload.csv', index_col=0)
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    except FileNotFoundError:
        st.warning("El archivo 'csv' no se encontró. Cargue un conjunto de datos en la opción Upload antes de ejecutar el perfilado.")

def modelling():
    st.subheader('Choose the supervised ML')
    try:
        df = pd.read_csv('./DS/ds_upload.csv', index_col=0)
        chosen_target = st.selectbox('selecciona', df.columns)
        features_for_regresion = list(df.select_dtypes(include=['float64']))
        features_for_clasification = list(df.select_dtypes(exclude=['float64']))

        if chosen_target in features_for_clasification:
            st.subheader('Clasificacion')
            if st.button('Run Preprocesing'):
                df = dummies(df)
                df = imputer(df)
                df = standardize(df)
                df = pd.read_csv('./DS/ds_preprocesing.csv', index_col=0)
                train_set, val_set, test_set = split(df)
                X_train, y_train, X_val, y_val, X_test, y_test = remove_labels(
                    train_set, val_set, test_set, chosen_target)
                X_train, y_train, X_val, y_val, X_test, y_test =  balancing(
                    X_train, y_train, X_val, y_val, X_test, y_test)
                st.dataframe(X_train)
                st.dataframe(y_train)
                df_classifiers = search_model(X_train, y_train, X_val, y_val, X_test, y_test, 1)
                st.dataframe(df_classifiers)
    except FileNotFoundError:
        st.warning("El archivo 'csv' no se encontró. Cargue un conjunto de datos en la opción Upload antes de ejecutar el perfilado.")

def sideBar():
    with st.sidebar:
        selected=option_menu(
            menu_title="Menu",
            options=["Info","Upload","Profiling", "Modelling", "Dashboard"],
            icons=["info", "archive", "menu-up", "menu-button", "eye"],
            menu_icon="cast",
            default_index=0
        )

    if selected=="Info":
        info()

    if selected=="Upload":
        upload()

    if selected=="Profiling":
        profiling()

    if selected=="Modelling":
        modelling()

    if selected=="Dashboard":
        dashboard()


if __name__ == "__main__":
    sideBar()