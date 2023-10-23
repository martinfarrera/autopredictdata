# Data Manipulation.
import pandas as pd

# DS Division.
from sklearn.model_selection import train_test_split

# Precision Metrics.
from sklearn.metrics import f1_score

# Warnings.
import warnings
warnings.filterwarnings("ignore")

# Time.
import time


# 1. Preprocessing Model
def preprocessing_data(df):
    # Get the features of type object.
    cat_obj = df.select_dtypes(include='object').columns
    # Object type Features are Coded.
    df = pd.get_dummies(df, columns=cat_obj, prefix_sep='-', drop_first=True)
    # Imputer Data
    df_values = KNNImputer(n_neighbors=4, weights="uniform").fit_transform(df)
    # New DF
    df = pd.DataFrame(df_values, columns=df.columns)
    return df


# 2. Divide the DF into the set of Training, Validation and Test.
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
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


# 4. Search Model
def search_model(names, models, X_train, y_train, X_val, y_val, X_test, y_test, pos_label):
    f1_Vali = []
    f1_Test = []
    fitting = []
    times = []

    for clf in models:
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

    df_models = pd.DataFrame([f1_Vali, f1_Test, fitting, times], columns=names,
                             index=['F1 Vali', 'F1 Test', 'Fitting', 'Seconds'])
    df_models = df_models.sort_values(by=['F1 Vali', 'F1 Test', 'Seconds', 'Fitting'], axis=1, ascending=False)

    return df_models