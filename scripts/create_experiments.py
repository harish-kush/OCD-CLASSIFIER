import mlflow

TRACKING_URI = "file:./mlruns"

mlflow.set_tracking_uri(TRACKING_URI)

experiments = [
    "Logistic Regression",
    "Random Forest",
    "LSTM",
    "BiLSTM"
]

for exp_name in experiments:
    exp = mlflow.get_experiment_by_name(exp_name)

    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
        print(f"Created: {exp_name} ({exp_id})")
    else:
        print(f"Exists: {exp_name} ({exp.experiment_id})")