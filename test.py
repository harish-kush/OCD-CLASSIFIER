import joblib

lr = joblib.load("models/logistic_regression.pkl")
rf = joblib.load("models/random_forest.pkl")

print(type(lr))
print(type(rf))
print("LR features:", lr.n_features_in_)
print("RF features:", rf.n_features_in_)