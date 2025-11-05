def train_model(data, model, params):
    """
    Train the specified model using the provided data and parameters.
    
    Parameters:
    - data: The training data.
    - model: The machine learning model to train.
    - params: A dictionary of hyperparameters for the model.
    
    Returns:
    - trained_model: The trained machine learning model.
    """
    model.set_params(**params)
    trained_model = model.fit(data['X_train'], data['y_train'])
    return trained_model

def evaluate_model(model, data):
    """
    Evaluate the trained model using the test data.
    
    Parameters:
    - model: The trained machine learning model.
    - data: The test data.
    
    Returns:
    - evaluation_metrics: A dictionary containing evaluation metrics.
    """
    predictions = model.predict(data['X_test'])
    evaluation_metrics = {
        'accuracy': accuracy_score(data['y_test'], predictions),
        'f1_score': f1_score(data['y_test'], predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(data['y_test'], predictions)
    }
    return evaluation_metrics

def save_model(model, filepath):
    """
    Save the trained model to a file.
    
    Parameters:
    - model: The trained machine learning model.
    - filepath: The path where the model should be saved.
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a trained model from a file.
    
    Parameters:
    - filepath: The path from where the model should be loaded.
    
    Returns:
    - model: The loaded machine learning model.
    """
    return joblib.load(filepath)