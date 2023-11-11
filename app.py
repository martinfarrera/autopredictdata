import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from fuctions import *
import models
import fuctions

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
        st.markdown("Mira más información sobre el DataSet que subiste:")
        st.dataframe(df_revision(df))
        st.markdown("##")
        st.markdown("---")

    st.markdown("##")
    st.markdown("##")

    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.subheader("Dataset de Prueba")
        st.markdown("Prueba la app con este DS sobre pacientes con acidentes cerebrovasculares.")
        df_test = pd.read_csv('./resources/healthcare-stroke-data.csv', index_col=0)
    with col2:
        st.markdown("##")
        if st.button('Usa este DataSet'):
            df_test.to_csv('./DS/ds_upload.csv')
    st.dataframe(df_test)
    st.markdown("Mira más información sobre el DataSet de pruebas:")
    st.dataframe(df_revision(df_test))

def profiling():
    st.subheader("Exploratory Data Analysis")

    try:
        df = pd.read_csv('./DS/ds_upload.csv', index_col=0)
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    except FileNotFoundError:
        st.warning("El archivo 'csv' no se encontró. Cargue un conjunto de datos en la opción Upload antes de ejecutar el perfilado.")

def modelling():
    st.subheader('Revisión de Datos')
    st.markdown('Mira las caracteristicas del DS que usaras para crear tu modelo.')

    try:
        df = pd.read_csv('./DS/ds_upload.csv', index_col=0)
        df = df.reset_index()
        st.dataframe(df_revision(df))

        st.markdown("##")
        col1, col2 = st.columns(2, gap='large')

        with col1:
            st.markdown("##")
            st.subheader('Preprocesamiento de Datos')

            showData = st.multiselect('Columnas a eliminar (si no hay nada por eliminar deja vacio): ', df.columns)
            chosen_prepro = st.multiselect('Elige los pasos que quieres usar en el preprocesamiento: ', ['Codificación','Llenar Vacios', 'Estandarización'])

            if st.button('Hacerlo'):
                df = df.drop(df[showData], axis=1)
                st.info("Se eliminaron las columnas seleccionadas")

                preprocesing(df, chosen_prepro)

            df = pd.read_csv('./DS/ds_preprosesing.csv', index_col=0)

            train_set, val_set, test_set = split(df)
            train_set, val_set, test_set = unnamed_col(train_set, val_set, test_set)

            st.markdown("##")
            st.subheader('Predicción')
            chosen_target = st.selectbox('Selecciona el dato que quieres predecir: ', train_set.columns)
            features_for_clasification = list(train_set.select_dtypes(exclude=['float64']).columns)

            if chosen_target in features_for_clasification:
                chosen_positive_label = st.selectbox('Selecciona el valor positivo: ', df[chosen_target].unique())

            if st.button('Crear Modelo'):
                with col2:
                    if chosen_target in features_for_clasification:
                        st.markdown("##")
                        st.subheader('Estos son los mejores modelos')

                        X_train, y_train, X_val, y_val, X_test, y_test = remove_labels(train_set, val_set, test_set, chosen_target)
                        X_train, y_train, X_val, y_val, X_test, y_test = balancing(X_train, y_train, X_val, y_val, X_test, y_test)

                        df_models_cl = search_model_cl(X_train, y_train, X_val, y_val, X_test, y_test, chosen_positive_label)
                        st.dataframe(df_models_cl)

                    if chosen_target not in features_for_clasification:
                        st.markdown("##")
                        st.subheader('El modelo que necesitas es de Regresión')

                        X_train, y_train, X_val, y_val, X_test, y_test = remove_labels(train_set, val_set, test_set, chosen_target)

                        df_models_rg = search_model_rg(X_train, y_train, X_val, y_val, X_test, y_test)
                        st.dataframe(df_models_rg)

    except FileNotFoundError:
        pass

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