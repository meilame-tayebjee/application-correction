import os
import yaml
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def import_yaml_config(filename: str) -> dict:
    """Import configuration from YAML file

    Args:
        filename (str, optional): _description_, .yaml file

    Returns:
        dict: _description_
    """
    dict_config = {}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as stream:
            dict_config = yaml.safe_load(stream)
    else:
        print("File not found, returning empty dict.")
        dict_config = {}
    return dict_config


def process_data(data_path):
    TrainingData = pd.read_csv(data_path)

    TrainingData["Ticket"].str.split("/").str.len()
    TrainingData["Name"].str.split(",").str.len()

    TrainingData.isnull().sum()

    # Statut socioéconomique
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
        "fréquence des Pclass"
    )
    sns.barplot(data=TrainingData, x="Pclass", y="Survived", ax=axes[1]).set_title(
        "survie des Pclass"
    )

    # Age
    sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
        "Distribution de l'âge"
    )
    plt.show()

    return TrainingData
