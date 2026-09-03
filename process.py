import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_data(raw: pd.DataFrame) -> tuple:
    """
    Preprocess the raw data to extract features and targets.

    Parameters:
    - raw: pd.DataFrame, the raw dataset.

    Returns:
    - tuple containing:
        - features_names: list of feature column names.
        - target_names: name of the target column.
        - raw_features: pd.DataFrame of features.
        - targets: np.array of target values.
    """
    features_names = raw.columns.tolist()[:-1]
    target_names = raw.columns.tolist()[-1]
    raw_features = raw[features_names]
    targets = np.array(raw[target_names].squeeze().tolist())

    return features_names, target_names, raw_features, targets


def create_features_targets_dict(raw_features: pd.DataFrame, targets: np.array, features_names: list, target_names: str) -> dict:
    """
    Create a dictionary containing features, targets, and their names.

    Parameters:
    - raw_features: pd.DataFrame of features.
    - targets: np.array of target values.
    - features_names: list of feature column names.
    - target_names: name of the target column.

    Returns:
    - dict containing 'data', 'target', 'feature_names', and 'target_names'.
    """
    features = []

    for name in features_names:
        f = []

        for feature in raw_features[name]:
            if isinstance(feature, (int, float)):
                f.append(feature)

        if f != []:
            features.append(f)
    features = np.array(features).T
    return {'data': features, 'target': targets, 'feature_names': features_names, 'target_names': target_names}


def process_data(data: dict) -> tuple:
    """
    Split the data into training and testing sets.

    Parameters:
    - data: dict containing 'data' and 'target'.

    Returns:
    - tuple containing:
        - X_train: training features.
        - X_test: testing features.
        - y_train: training targets.
        - y_test: testing targets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        data['data'], data['target'], test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test