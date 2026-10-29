import kagglehub
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
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
print(df["SaleType"].unique())


def objective_fn(trial: optuna.Trial):

    strategy = trial.suggest_categorical("trial", ["mean", "median", "most_frequent"])
    imputer = SimpleImputer()
