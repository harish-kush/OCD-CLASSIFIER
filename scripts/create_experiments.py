import dagshub
import mlflow

# DagsHub MLflow Tracking
dagshub.init(
    repo_owner="harish-kush",
    repo_name="OCD-CLASSIFIER",
    mlflow=True
)

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