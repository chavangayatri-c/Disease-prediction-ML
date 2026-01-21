from src.preprocess import load_and_preprocess
from src.train_models import get_models
from src.evaluate import evaluate

# -------- HEART DISEASE --------
print("===== HEART DISEASE PREDICTION =====")
X_train, X_test, y_train, y_test = load_and_preprocess(
    "data/heart.csv", "target"
)

models = get_models()
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test)

# -------- DIABETES --------
print("\n===== DIABETES PREDICTION =====")
X_train, X_test, y_train, y_test = load_and_preprocess(
    "data/diabetes.csv", "Outcome"
)

for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test)

# -------- BREAST CANCER --------
print("\n===== BREAST CANCER PREDICTION =====")
X_train, X_test, y_train, y_test = load_and_preprocess(
    "data/breast_cancer.csv", "diagnosis"
)

for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test)
