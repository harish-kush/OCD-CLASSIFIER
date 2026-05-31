# OCD vs Non-OCD Classification using EEG Signals

## Overview

This project presents an EEG-based classification system for detecting Obsessive-Compulsive Disorder (OCD) using Machine Learning and Deep Learning models. The objective is to analyze EEG signals and classify subjects as OCD or Non-OCD.

The project compares traditional machine learning approaches with sequence-based deep learning architectures and uses MLflow for experiment tracking, model management, and performance comparison.

---

## Features

* EEG Signal Classification
* Logistic Regression Implementation
* Random Forest Implementation
* LSTM-based Deep Learning Model
* BiLSTM-based Deep Learning Model
* MLflow Experiment Tracking
* Model Registry Integration
* Automatic Metrics Logging
* ROC, PR Curve and Confusion Matrix Generation

---

## Project Structure

```text
OCD VS NON OCD CLASSIFIER
│
├── data/
│   ├── X_test_aug.npy
│   └── y_test_aug.npy
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── lstm_model.h5
│   └── bilstm_new.h5
│
├── artifacts/
│
├── mlruns/
│
├── scripts/
│   ├── create_experiments.py
│   ├── generate_metrics.py
│   ├── register_model.py
│   └── launch_mlflow.py
│
└── requirements.txt
```

---

## Models Used

### Logistic Regression

A baseline machine learning model trained using handcrafted statistical EEG features.

### Random Forest

An ensemble learning approach using multiple decision trees for classification.

### LSTM

A recurrent neural network architecture capable of learning temporal dependencies in EEG sequences.

### BiLSTM

A Bidirectional LSTM model that processes EEG sequences in both forward and backward directions, improving contextual understanding of EEG patterns.

---

## Feature Engineering

For Logistic Regression and Random Forest, the following features were extracted:

* Mean
* Standard Deviation
* Maximum Value
* Minimum Value
* Signal Energy

---

## Results

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     49.91% |     39.56% |     47.80% |     43.29% |      0.495 |
| Random Forest       |     66.17% |     94.01% |     16.48% |     28.04% |      0.624 |
| LSTM                |     90.11% |     83.42% |     93.95% |     88.37% |     0.9573 |
| **BiLSTM**          | **91.00%** | **85.63%** | **90.58%** | **88.03%** | **0.9565** |

---

## Best Model

### BiLSTM

The BiLSTM model achieved the best overall performance and was selected as the final production model.

**Performance:**

* Accuracy: 91.0%
* Precision: 85.63%
* Recall: 90.58%
* F1 Score: 88.03%
* ROC-AUC: 0.9565

The model was registered in the MLflow Model Registry for version control and deployment readiness.

---

## MLflow Integration

The project uses MLflow for:

* Experiment Tracking
* Model Comparison
* Metrics Logging
* Artifact Management
* Model Registry

### Logged Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Logged Artifacts

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

---

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Create Experiments

```bash
python scripts/create_experiments.py
```

### Evaluate Models and Log Results

```bash
python scripts/generate_metrics.py
```

### Launch MLflow UI

```bash
python scripts/launch_mlflow.py
```

Open:

```text
http://localhost:5000
```

---

## Technologies Used

* Python
* NumPy
* Scikit-Learn
* TensorFlow
* Keras
* Matplotlib
* Seaborn
* MLflow

---

## Future Work

* Real-Time EEG Classification
* Streamlit Dashboard Deployment
* Docker Support
* Cloud Deployment
* Hyperparameter Optimization
* Cross-Subject Generalization

---

## Author

**Harish Kushwaha**

Electronics and Communication Engineering (ECE)
MANIT Bhopal

---

## License

This project is intended for academic and research purposes.
