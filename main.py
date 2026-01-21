# 1️⃣ Preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess("data/heart.csv", "target")

# 2️⃣ Get models
models = get_models()

# 3️⃣ Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test)
