"""
Prediction de la survie d'un individu sur le Titanic
"""
from pathlib import Path
import argparse
import joblib

from titanicml.data.import_data import import_yaml_config, process_data
from titanicml.pipeline.build_pipeline import split, build_pipeline
from titanicml.models.train_evaluate import evaluate
from sklearn.model_selection import GridSearchCV
from src.models import log as mlog

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument("--n_trees", type=int, default=20, help="Nombre d'arbres")
parser.add_argument("--appli", type=str, default="appli21", help="Application number")

args = parser.parse_args()

N_TREES = args.n_trees
print("Nombre d'arbres : ", N_TREES)

config = import_yaml_config("configuration/config.yaml")

DATA_PATH = config.get("data_path", "data.csv")
TRAIN_PATH = config.get("train_path", "train.csv")
TEST_PATH = config.get("test_path", "test.csv")
TEST_FRACTION = config.get("test_fraction", 0.1)
MAX_DEPTH = None
MAX_FEATURES = "sqrt"
EXPERIMENT_NAME = "titanicml"
APPLI_ID = args.appli


# IMPORT ET EXPLORATION DONNEES --------------------------------

TrainingData = process_data(DATA_PATH)

# SPLIT TRAIN/TEST --------------------------------

X_train, X_test, y_train, y_test = split(
    TrainingData,
    test_fraction=TEST_FRACTION,
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
)

def log_local_data(data, filename):
    data.to_csv(f"data/intermediate/{filename}.csv", index=False)


output_dir = Path("data/intermediate")
output_dir.mkdir(parents=True, exist_ok=True)

log_local_data(X_train, "X_train")
log_local_data(X_test, "X_test")
log_local_data(y_train, "y_train")
log_local_data(y_test, "y_test")
# PIPELINE ----------------------------

# Définition des variables
numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

pipe = build_pipeline(
    numeric_features,
    categorical_features,
    n_trees=N_TREES,
    max_depth=MAX_DEPTH,
    max_features=MAX_FEATURES,
)

# ESTIMATION ET EVALUATION ----------------------

pipe.fit(X_train, y_train)


param_grid = {
    "classifier__n_estimators": [10, 20, 50],
    "classifier__max_leaf_nodes": [5, 10, 50],
}
pipe_cross_validation = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring=["accuracy", "precision", "recall", "f1"],
    refit="f1",
    cv=5,
    n_jobs=5,
    verbose=1,
)

pipe_cross_validation.fit(X_train, y_train)
pipe = pipe_cross_validation.best_estimator_

joblib.dump(pipe, "model.joblib")
mlog.log_gsvc_to_mlflow(pipe_cross_validation, EXPERIMENT_NAME, APPLI_ID)