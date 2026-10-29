import kagglehub
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
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
    df, test_size=0.2, train_size=0.8, random_state=42, shuffle=True
)


def objective_fn(trial: optuna.Trial):

    cat_cols = [col for col in df.columns() if col not in ["int64", "float64"]]
    cat_df = df[cat_cols]
    cat_imputer = SimpleImputer(strategy="most_frequent")

    multi_binarizer = MultiLabelBinarizer()
    multi_binarizer.fit(cat_df)
    cat_df = multi_binarizer.transform(cat_df)

    num_strategy = trial.suggest_categorical(
        "trial", ["mean", "median", "most_frequent"]
    )
    num_imputer = SimpleImputer(strategy=num_strategy)

    num_df = df[[col for col in df.columns if df[col].dtype in ["int64", "float64"]]]
    num_imputer.fit(num_df)
    num_df = num_imputer.transform(num_df)
