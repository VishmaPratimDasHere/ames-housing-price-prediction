import kagglehub
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
import pandas as pd
import pathlib
import optuna

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


def objective_fn(trial: optuna.Trial):

    # dealing with categorical values

    # step 1: impute
    cat_cols = [
        col for col in df_train.columns if df[[col]].dtype not in ["int64", "float64"]
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
        "trial", ["mean", "median", "most_frequent"]
    )
    num_imputer = SimpleImputer(strategy=num_strategy)

    num_df: pd.DataFrame = df_train[
        [col for col in df_train.columns if df_train[col].dtype in ["int64", "float64"]]
    ]
    num_df = num_imputer.fit_transform(num_df)

    new_df = num_df.merge(right=cat_df, how="outer")

    # Training the model
    model = LinearRegression(n_jobs=5)
    kfold = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(
        model,
        X=new_df[[col for col in new_df.columns if not col == "SalePrice"]],
        y=new_df["SalePrice"],
        cv=kfold,
    )
    print(scores.mean())
