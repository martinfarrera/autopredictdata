
import pandas as pd # Data Manipulation
import numpy as np
from sklearn.impute import KNNImputer # Imputer
from sklearn.preprocessing import RobustScaler # Data Standardization.
from sklearn.model_selection import train_test_split # DS Division
from sklearn.metrics import f1_score # Precision Metrics
from sklearn.metrics import r2_score
from imblearn.over_sampling import SMOTE # Balancing
import warnings # Warnings.
warnings.filterwarnings("ignore")
import time # Time.
import models # models.Py


def df_revision(df):
    col_values_count = {i: df[i].value_counts().shape[0] for i in df.columns}
    col_na = {i: ('🔸' if df[i].isna().any() == True else '-') for i in df.columns}
    col_isObj = {i:('🔹' if df[i].dtype == 'object' else '-') for i in df.columns}
    col_isStand = {i: ('🔹' if df[i].dtype == 'int' else '-') for i in df.columns}
    col_dtype = {i: df[i].dtype for i in df.columns}

    # Agregar una columna con el porcentaje de valores faltantes
    col_porcentaje_faltante = {
        i: f'{(df[i].isna().sum() / len(df)) * 100:.2f}% - {"🔸" if df[i].isna().any() else "-"}' if df[i].isna().any() else '-'
        for i in df.columns
    }

    # Agregar una columna 'balanceo' solo para las variables 'object' e 'int'
    col_balanceo = {
        i: '🔸 ' if df[i].dtype in ['object', 'int'] and len(df[i].value_counts()) > 1 and (df[i].value_counts() / len(df)).min() < 0.25 else '-'
        for i in df.columns
    }

    df_revision = pd.DataFrame([col_porcentaje_faltante, col_isObj, col_isStand, col_values_count, col_balanceo], index=['% Faltantes', 'Codificables', 'Estandarizables', 'Valores Unicos', 'Desbalanceados']).transpose()
    return df_revision

def convert_types(df):
    cat_obj = df.select_dtypes(include=['string','bool','category']).columns
    df[cat_obj] = df[cat_obj].astype('object')
    cat_int = df.select_dtypes(include=['int']).columns
    df[cat_int] = df[cat_int].astype('int64')
    cat_float = df.select_dtypes(include=['float']).columns
    df[cat_float] = df[cat_float].astype('float64')
    return df

def unnamed_col(train_set, val_set, test_set):
    # Verificar si la columna "Unnamed: 0" está presente y eliminarla en cada DataFrame
    for df in [train_set, val_set, test_set]:
        if 'Unnamed: 0' in df.columns:
            # Eliminar la columna "Unnamed: 0"
            df.drop('Unnamed: 0', axis=1, inplace=True)

    return train_set, val_set, test_set


def preprocesing(df, chosen_prepro):

    pre_functions = {
        'Codificación': encoding,
        'Llenar Vacios': imputer,
        'Estandarización': standardize
    }

    for ftc_name in chosen_prepro:
        df = pre_functions[ftc_name](df)
    df.to_csv('./DS/ds_preprosesing.csv')
    return df


def encoding(df):
    cat_obj = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_obj, prefix_sep=' - ')

    # Convertir columnas booleanas en tipo int64
    for feature in df.columns:
        if df[feature].dtype == bool:
            df[feature] = df[feature].astype(int)

    return df

def imputer(df):
    original_dtypes = df.dtypes.to_dict()
    df_values = KNNImputer(n_neighbors=4, weights="uniform").fit_transform(df)
    df = pd.DataFrame(df_values, columns=df.columns)
    df = df.astype(original_dtypes)
    return df

def standardize(df):
    c_features = list(df.select_dtypes(include=['float64']).columns)
    df[c_features] = RobustScaler().fit_transform(df[c_features])
    return df

# 2. Divide the DF into the set of Training, Validation and Test.
def split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)

    return train_set, val_set, test_set


# 3. Remove Target (Separates the Label from the Features)
def remove_labels(train_set, val_set, test_set, target_name):
    X_train = train_set.drop(target_name, axis=1)
    y_train = train_set[target_name].copy()
    X_val = val_set.drop(target_name, axis=1)
    y_val = val_set[target_name].copy()
    X_test = test_set.drop(target_name, axis=1)
    y_test = test_set[target_name].copy()
    return X_train, y_train, X_val, y_val, X_test, y_test

def balancing(X_train, y_train, X_val, y_val, X_test, y_test):
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_val, y_val = smote.fit_resample(X_val, y_val)
    X_test, y_test = smote.fit_resample(X_test, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test

# 4. Search Model
def search_model_cl(X_train, y_train, X_val, y_val, X_test, y_test, pos_label, cnames=models.cnames, classifiers=models.classifiers):
    f1_Vali = []
    f1_Test = []
    fitting = []
    times = []

    for clf in classifiers:
        # ======= TRAIN ========
        start = time.time()
        clf.fit(X_train, y_train.values.ravel())
        end = time.time()
        times.append(round(end - start, 3))

        # ===== VALIDATION =====
        y_val_pred = clf.predict(X_val)
        f1_val = f1_score(y_val, y_val_pred, pos_label=pos_label)

        # ======= TEST ========
        y_test_pred = clf.predict(X_test)
        f1_test = f1_score(y_test, y_test_pred, pos_label=pos_label)

        # ======= SCORE =======
        f1_Vali.append(round(f1_val * 100, 3))
        f1_Test.append(round(f1_test * 100, 3))
        fitting.append(round((f1_val - f1_test) * 100, 3))

    df_models = pd.DataFrame([f1_Vali, f1_Test, fitting, times], columns=cnames,
                             index=['F1 Vali', 'F1 Test', 'Fitting', 'Seconds'])
    df_models = df_models.sort_values(by=['F1 Vali', 'F1 Test', 'Seconds', 'Fitting'], axis=1, ascending=False)

    return df_models

def search_model_rg(X_train, y_train, X_val, y_val, X_test, y_test, rnames=models.rnames, regressors=models.regressors):
    r2_Vali = []
    r2_Test = []
    fitting = []
    times = []

    for reg in regressors:
        # ======= TRAIN ========
        start = time.time()
        reg.fit(X_train, y_train.values.ravel())
        end = time.time()
        times.append(round(end - start, 3))

        # ===== VALIDATION =====
        y_val_pred = reg.predict(X_val)
        r2_val = r2_score(y_val, y_val_pred)

        # ======= TEST ========
        y_test_pred = reg.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)

        # ======= SCORE =======
        r2_Vali.append(round(r2_val, 3))
        r2_Test.append(round(r2_test, 3))
        fitting.append(round((r2_val - r2_test), 3))

    df_models_rg = pd.DataFrame([r2_Vali, r2_Test, fitting, times], columns=rnames,
                             index=['R2 Vali', 'R2 Test', 'Fitting', 'Seconds'])
    df_models_rg = df_models_rg.sort_values(by=['R2 Vali', 'R2 Test', 'Seconds', 'Fitting'], axis=1, ascending=False)

    return df_models_rg