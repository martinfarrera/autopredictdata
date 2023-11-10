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