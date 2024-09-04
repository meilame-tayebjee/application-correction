import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def split(
    TrainingData,
    test_fraction: float = 0.1,
    train_path: str = "train.csv",
    test_path: str = "test.csv",
):
    from sklearn.model_selection import train_test_split

    y = TrainingData["Survived"]
    X = TrainingData.drop("Survived", axis="columns")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)
    pd.concat([X_train, y_train]).to_csv(train_path)
    pd.concat([X_test, y_test]).to_csv(test_path)

    return X_train, X_test, y_train, y_test


def build_pipeline(
    numeric_features,
    categorical_features,
    n_trees: int,
    max_depth: int,
    max_features: int,
):

    # Variables numériques
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Variables catégorielles
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, numeric_features),
            (
                "Preprocessing categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    # Pipeline
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_trees, max_depth=max_depth, max_features=max_features
                ),
            ),
        ]
    )

    return pipe
