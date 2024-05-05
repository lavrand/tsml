import argparse
import os
import time
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestCentroid
from tslearn.datasets import UCR_UEA_datasets
from tslearn.metrics import dtw, soft_dtw

def run_ncc_experiment(dataset_name, metric='euclidean', gamma=None):
    # Load the dataset
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

    # Initialize the model with a default
    model = None

    # Initialize the model
    if metric == 'euclidean':
        model = NearestCentroid()
    elif metric == 'dtw':
        model = NearestCentroid(metric=dtw)
    elif metric == 'softdtw' and gamma is not None:
        model = NearestCentroid(metric=lambda x, y: soft_dtw(x, y, gamma=gamma))

    # Check if model is still None
    if model is None:
        raise ValueError(f"Invalid metric: {metric} or gamma: {gamma}")

    # Get initial RAM usage
    process = psutil.Process(os.getpid())
    initial_ram_usage = process.memory_info().rss  # in bytes

    # Fit the model and measure the time
    start_fit = time.time()
    model.fit(X_train, y_train)
    end_fit = time.time()

    # Predict the labels and measure the time
    start_predict = time.time()
    y_pred = model.predict(X_test)
    end_predict = time.time()

    # Get final RAM usage
    final_ram_usage = process.memory_info().rss  # in bytes

    # Calculate RAM usage in GB
    ram_usage = (final_ram_usage - initial_ram_usage) / (1024 ** 3)

    # Calculate the metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Calculate the timings
    fit_time = end_fit - start_fit
    predict_time = end_predict - start_predict
    total_time = fit_time + predict_time

    # Return the metrics and timings
    return {
        'Dataset': dataset_name,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Fit Time': fit_time,
        'Predict Time': predict_time,
        'Total Time': total_time,
        'RAM Usage (GB)': ram_usage
    }

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Run NCC experiment with specified parameters.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use.')
    parser.add_argument('--metric', type=str, required=True, help='Distance metric to use.')
    parser.add_argument('--gamma', type=float, required=False, help='Gamma value for SoftDTW metric.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the NCC experiment with the specified parameters
    result = run_ncc_experiment(args.dataset, args.metric, args.gamma)
    print(result)