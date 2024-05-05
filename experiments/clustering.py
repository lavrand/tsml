import argparse
import os
import time
import pandas as pd
import psutil
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, \
    homogeneity_score, completeness_score, v_measure_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import UCR_UEA_datasets
from tslearn.metrics import dtw, soft_dtw

def run_clustering_experiment(dataset_name, n_clusters, metric='euclidean', gamma=None):
    # Load the dataset
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

    # Initialize the model with a default
    model = None

    if metric == 'euclidean':
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric)
    elif metric == 'dtw':
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric=dtw)
    elif metric == 'softdtw' and gamma is not None:
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric=lambda x, y: soft_dtw(x, y, gamma=gamma))

    # Check if model is still None
    if model is None:
        raise ValueError(f"Invalid metric: {metric} or gamma: {gamma}")

    # Get initial RAM usage
    process = psutil.Process(os.getpid())
    initial_ram_usage = process.memory_info().rss  # in bytes

    # Fit the model and measure the time
    start_fit = time.time()
    model.fit(X_train)
    end_fit = time.time()

    # Predict the labels for X_train and measure the time
    start_predict_train = time.time()
    y_pred_train = model.predict(X_train)
    end_predict_train = time.time()

    # Predict the labels for X_test and measure the time
    start_predict_test = time.time()
    y_pred_test = model.predict(X_test)
    end_predict_test = time.time()

    # Get final RAM usage
    final_ram_usage = process.memory_info().rss  # in bytes

    # Calculate RAM usage in GB
    ram_usage = (final_ram_usage - initial_ram_usage) / (1024 ** 3)

    # Calculate the metrics for X_train
    ari_train = adjusted_rand_score(y_train, y_pred_train)
    nmi_train = normalized_mutual_info_score(y_train, y_pred_train)
    ami_train = adjusted_mutual_info_score(y_train, y_pred_train)
    homogeneity_train = homogeneity_score(y_train, y_pred_train)
    completeness_train = completeness_score(y_train, y_pred_train)
    v_score_train = v_measure_score(y_train, y_pred_train)

    # Calculate the metrics for X_test
    ari_test = adjusted_rand_score(y_test, y_pred_test)
    nmi_test = normalized_mutual_info_score(y_test, y_pred_test)
    ami_test = adjusted_mutual_info_score(y_test, y_pred_test)
    homogeneity_test = homogeneity_score(y_test, y_pred_test)
    completeness_test = completeness_score(y_test, y_pred_test)
    v_score_test = v_measure_score(y_test, y_pred_test)

    # Calculate the timings
    fit_time = end_fit - start_fit
    predict_time_train = end_predict_train - start_predict_train
    predict_time_test = end_predict_test - start_predict_test
    total_time = fit_time + predict_time_train + predict_time_test

    # Return the metrics and timings
    return {
        'Dataset': dataset_name,
        'ARI Train': ari_train,
        'NMI Train': nmi_train,
        'AMI Train': ami_train,
        'Homogeneity Train': homogeneity_train,
        'Completeness Train': completeness_train,
        'V-score Train': v_score_train,
        'ARI Test': ari_test,
        'NMI Test': nmi_test,
        'AMI Test': ami_test,
        'Homogeneity Test': homogeneity_test,
        'Completeness Test': completeness_test,
        'V-score Test': v_score_test,
        'Fit Time': fit_time,
        'Predict Time Train': predict_time_train,
        'Predict Time Test': predict_time_test,
        'Total Time': total_time,
        'RAM Usage (GB)': ram_usage
    }

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Run clustering experiment with specified parameters.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use.')
    parser.add_argument('--n_clusters', type=int, required=True, help='Number of clusters to use.')
    parser.add_argument('--metric', type=str, required=True, help='Distance metric to use.')
    parser.add_argument('--gamma', type=float, required=False, help='Gamma value for SoftDTW metric.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the clustering experiment with the specified parameters
    result = run_clustering_experiment(args.dataset, args.n_clusters, args.metric, args.gamma)
    print(result)