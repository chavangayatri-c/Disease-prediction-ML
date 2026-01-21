from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, X_test, y_test):
    """
    Evaluates a trained ML model and prints accuracy and classification report
    """
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
