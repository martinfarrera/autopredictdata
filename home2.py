import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from fuctions import *

st.set_page_config(page_title="Auto Data Predict",page_icon="🔮️",layout="wide")
st.title("⚡️ Auto Data Predict")
st.markdown("##")

st.sidebar.image("utils/logo.png")
st.sidebar.markdown("##")

# Declarar df en un alcance global
df = None
profile_df = None  # Declarar profile_df en un alcance global

def info():
    st.subheader("Descripción")
    st.markdown("Un programa de machine learning predice datos mediante la recopilación de información histórica y variables relevantes. Los datos se procesan y se entrena un modelo con un algoritmo adecuado. Luego, se evalúa su rendimiento, se ajusta si es necesario y se utiliza para hacer predicciones futuras en un entorno de producción.")

    st.subheader("Dataset de Prueba")
    global df  # Accede a la variable global df
    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=0)
        st.dataframe(df)

def upload():
    st.title('Upload Your Data Set (file_name.csv)')
    file = st.file_uploader('Upload DS')
    if file:
        global df  # Accede a la variable global df
        df = pd.read_csv(file, index_col=0)
        df.to_csv('dataset.csv')
        st.dataframe(df)

def profiling():
    st.title('Upload Your Data Set (file_name.csv)')
    file = st.file_uploader('Upload DS')
    if file:
        global df  # Accede a la variable global df
        df = pd.read_csv(file, index_col=0)
        df.to_csv('dataset.csv')
        st.dataframe(df)

    st.title("Exploratory Data Analysis")
    global profile_df

    if df is not None:
        if st.button('Run Profiling'):
            profile_df = ProfileReport(df)
            st.write("Profile Report Generado. Puedes verlo a continuación.")
        if profile_df is not None:
            st.subheader("Profile Report")
            st_profile_report(profile_df)
    else:
        st.warning("Cargue un conjunto de datos antes de ejecutar el perfilado.")


def sideBar():
    with st.sidebar:
        selected=option_menu(
            menu_title="Menu",
            options=["Info","Upload","Profiling", "Modelling", "Dashboard"],
            icons=["info","archive", "menu-up", "menu-button", "eye"],
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


def balancing(X_train, y_train, X_val, y_val, X_test, y_test):
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_val, y_val = smote.fit_resample(X_val, y_val)
    X_test, y_test = smote.fit_resample(X_test, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test