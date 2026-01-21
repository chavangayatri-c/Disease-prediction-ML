from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    """
    Returns a dictionary of machine learning models
    """
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }
    return models
