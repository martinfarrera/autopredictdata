
import pandas as pd # Data Manipulation
import numpy as np
from sklearn.impute import KNNImputer # Imputer
from sklearn.preprocessing import RobustScaler # Data Standardization.
from sklearn.model_selection import train_test_split # DS Division
from sklearn.metrics import f1_score # Precision Metrics
from imblearn.over_sampling import SMOTE # Balancing
import warnings # Warnings.
warnings.filterwarnings("ignore")
import time # Time.
import models # models.Py

def df_revision(df):
    col_values_count = {i: df[i].value_counts().shape[0] for i in df.columns}
    col_na = {i: ('🔴' if df[i].isna().any() == True else '-') for i in df.columns}
    col_isObj = {i:('🟢' if df[i].dtype in ['object', 'category', 'string', 'bool'] else '-') for i in df.columns}
    col_isStand = {i: ('🟢' if df[i].dtype in ['int', 'int64','float', 'float64'] else '-') for i in df.columns}
    col_dtype = {i: df[i].dtype for i in df.columns}
    df_revision = pd.DataFrame([col_values_count, col_na, col_isObj, col_isStand, col_dtype,], index=['Valores Unicos', 'Vacios', 'Codificables', 'Estandarizables', 'Tipo de Dato']).transpose()
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

# 1. Preprocessing Model
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
    c_features = ['age', 'avg_glucose_level', 'bmi']
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

    train_set.to_csv('./DS/train_set.csv')
    val_set.to_csv('./DS/val_set.csv')
    test_set.to_csv('./DS/test_set.csv')

    train_set = pd.read_csv('./DS/train_set.csv')
    val_set = pd.read_csv('./DS/val_set.csv')
    test_set = pd.read_csv('./DS/test_set.csv')

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
def search_model(X_train, y_train, X_val, y_val, X_test, y_test, pos_label, cnames=models.c_names, clssfrs=models.classifiers):
    f1_Vali = []
    f1_Test = []
    fitting = []
    times = []

    for clf in clssfrs:
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