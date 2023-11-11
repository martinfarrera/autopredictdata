preprocess_functions = {
    'Codificación': encoding,
    'Llenar Vacios': imputer,
    'Estandarización': standardize
}
st.dataframe(df)

chosen_prepro = st.multiselect('Seleciona los preprocesamientos: ', list(preprocess_functions.keys()))

if st.button('Preprocesamiento'):
    for ftc_name in chosen_prepro:
        preprocess_functions[ftc_name](df)
        df = df

st.dataframe(df)





=========


with col1:
    st.markdown("##")
    st.subheader('Personaliza tu modelo de Machine Learning')

    showData = st.multiselect('Seleciona la columna a eliminar: ', df.columns)
    if st.button('Eliminar Columnas'):
        df = df.drop(df[showData], axis=1)
        df.to_csv('./DS/ds_delete_features.csv')
        st.info("Se eliminaron las columnas seleccionadas")

    df = pd.read_csv('./DS/ds_delete_features.csv', index_col=0)

    preprocess_functions = {
        'Codificación': encoding,
        'Llenar Vacios': imputer,
        'Estandarización': standardize
    }

    chosen_prepro = st.multiselect('Seleciona los preprocesamientos: ', list(preprocess_functions.keys()))

    if st.button('Preprocesamiento'):
        for ftc_name in chosen_prepro:
            df = preprocess_functions[ftc_name](df)
        df.to_csv('./DS/ds_preprosesing.csv')

    df = pd.read_csv('./DS/ds_preprosesing.csv', index_col=0)

    train_set, val_set, test_set = split(df)
    train_set, val_set, test_set = unnamed_col(train_set, val_set, test_set)

    st.markdown("##")
    st.subheader('Clase a predecir')
    chosen_target = st.selectbox('Selecciona etiqueta a predecir: ', train_set.columns)
    features_for_clasification = list(train_set.select_dtypes(exclude=['float64']).columns)

    if chosen_target in features_for_clasification:
        chosen_positive_label = st.selectbox('Selecciona el valor positivo: ', df[chosen_target].unique())