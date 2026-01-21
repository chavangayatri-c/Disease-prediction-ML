import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path, target):
    """
    Loads CSV dataset, separates features and target,
    scales the features, and splits into train/test sets.
    
    Parameters:
        path (str): Path to the CSV dataset
        target (str): Name of the target column
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load data
    df = pd.read_csv(path)
    
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


