"""
Prediction de la survie d'un individu sur le Titanic
"""

import argparse

from src.data.import_data import import_yaml_config, process_data
from src.pipeline.build_pipeline import split, build_pipeline
from src.models.train_evaluate import evaluate

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument("--n_trees", type=int, default=20, help="Nombre d'arbres")
args = parser.parse_args()

N_TREES = args.n_trees

config = import_yaml_config("configuration/config.yaml")

API_TOKEN = config.get("jeton_api")
DATA_PATH = config.get("data_path", "data.csv")
TRAIN_PATH = config.get("train_path", "train.csv")
TEST_PATH = config.get("test_path", "test.csv")
TEST_FRACTION = config.get("test_fraction", 0.1)
MAX_DEPTH = None
MAX_FEATURES = "sqrt"


# IMPORT ET EXPLORATION DONNEES --------------------------------

TrainingData = process_data(DATA_PATH)

# SPLIT TRAIN/TEST --------------------------------

X_train, X_test, y_train, y_test = split(
    TrainingData,
    test_fraction=TEST_FRACTION,
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
)
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
evaluate(pipe, X_test, y_test)
