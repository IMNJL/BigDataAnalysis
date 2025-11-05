def create_features(data):
    """
    Create new features from the existing dataset.

    Parameters:
    data (DataFrame): The input dataset.

    Returns:
    DataFrame: The dataset with new features added.
    """
    # Example feature engineering
    data['new_feature'] = data['existing_feature'] * 2  # Replace with actual feature engineering logic
    return data

def encode_categorical_features(data, categorical_columns):
    """
    Encode categorical features using one-hot encoding.

    Parameters:
    data (DataFrame): The input dataset.
    categorical_columns (list): List of categorical columns to encode.

    Returns:
    DataFrame: The dataset with encoded categorical features.
    """
    return pd.get_dummies(data, columns=categorical_columns, drop_first=True)

def scale_numerical_features(data, numerical_columns, scaler):
    """
    Scale numerical features using the provided scaler.

    Parameters:
    data (DataFrame): The input dataset.
    numerical_columns (list): List of numerical columns to scale.
    scaler: Scikit-learn scaler instance.

    Returns:
    DataFrame: The dataset with scaled numerical features.
    """
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data