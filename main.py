import kagglehub
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pathlib
import optuna
from pathlib import Path


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.DataFrame,
    study: optuna.Study,
) -> int:
    try:
        cat_cols = [
            col
            for col in train_df.columns
            if train_df[col].dtype not in ["int64", "float64"]
        ]
        cat_df = train_df[cat_cols]
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_data = cat_imputer.fit_transform(cat_df)
        multi_binarizer = MultiLabelBinarizer()
        multi_binarizer.fit(cat_df)
        cat_df = multi_binarizer.transform(cat_df)

        num_strategy = study.best_params[""]
        num_imputer = SimpleImputer(strategy=num_strategy)
        num_df = df_train[
            [
                col
                for col in df_train.columns
                if df_train[col].dtype in ["int64", "float64"]
            ]
        ]
        num_df = num_imputer.fit_transform(num_df.to_numpy())

        train_df = np.concatenate((cat_df, num_df), axis=1)

        model = LinearRegression()

        model.fit(X=train_df, y=y_train)

        pd.DataFrame({f"{y_train.columns[0]}": model.predict(test_df)}).to_csv(
            path.cwd() / "y_predicted.csv"
        )
        return 1
    except Exception as e:
        raise Exception(f"Error encountered:\n{repr(e)}")


def objective_fn(trial: optuna.Trial) -> float:
    """
    Objective function for optuna hyperparameter tuning.
    Assumes all columns are either categorical (no ordinal) or numerical.

    Params:
        *trial*: trial object. Automatically passed by *study.optimise()* function.
    """

    # dealing with categorical values

    # step 1: impute
    cat_cols = [
        col
        for col in df_train.columns
        if df_train[col].dtype not in ["int64", "float64"]
    ]
    cat_df = df_train[cat_cols]
    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_df = cat_imputer.fit_transform(cat_df)

    # step 2: binarize
    multi_binarizer = MultiLabelBinarizer()
    multi_binarizer.fit(cat_df)
    cat_df = multi_binarizer.transform(cat_df)

    # dealing with numerical cols

    # step 1: impute
    num_strategy = trial.suggest_categorical(
        "num_strategy", ["mean", "median", "most_frequent"]
    )
    num_imputer = SimpleImputer(strategy=num_strategy)

    num_df = df_train[
        [col for col in df_train.columns if df_train[col].dtype in ["int64", "float64"]]
    ]
    num_df = num_imputer.fit_transform(num_df.to_numpy())

    new_df = np.hstack((num_df, cat_df))

    # Training the model
    model = LinearRegression(n_jobs=5)
    kfold = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(
        model,
        X=new_df,
        y=y_train,
        cv=kfold,
    )
    print(sum(scores) / len(scores))
    return sum(scores) / len(scores)


kagglehub.login()

path = pathlib.Path(
    kagglehub.competition_download(
        "house-prices-advanced-regression-techniques", force_download=True
    )
)

print("Path to competition files:", path)

df = pd.read_csv(path / "train.csv", index_col=None)
print(df.head())

df_train, df_test, y_train, y_test = train_test_split(
    df[[col for col in df.columns if not col == "SalePrice"]],
    df[["SalePrice"]],
    test_size=0.2,
    train_size=0.8,
    random_state=42,
    shuffle=True,
)


study = optuna.create_study(direction="maximize")
study.optimize(func=objective_fn, n_trials=10000, n_jobs=2, gc_after_trial=True)
print(study.best_params)

if train_and_eval(train_df=df_train, test_df=df_test, y_train=y_train, study=study):
    print("Done")
