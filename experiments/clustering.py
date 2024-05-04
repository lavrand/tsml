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

    # Predict the labels and measure the time
    start_predict = time.time()
    y_pred = model.predict(X_test)
    end_predict = time.time()

    # Get final RAM usage
    final_ram_usage = process.memory_info().rss  # in bytes

    # Calculate RAM usage in GB
    ram_usage = (final_ram_usage - initial_ram_usage) / (1024 ** 3)

    # Calculate the metrics
    ari = adjusted_rand_score(y_test, y_pred)
    nmi = normalized_mutual_info_score(y_test, y_pred)
    ami = adjusted_mutual_info_score(y_test, y_pred)
    homogeneity = homogeneity_score(y_test, y_pred)
    completeness = completeness_score(y_test, y_pred)
    v_score = v_measure_score(y_test, y_pred)

    # Calculate the timings
    fit_time = end_fit - start_fit
    predict_time = end_predict - start_predict
    total_time = fit_time + predict_time

    # Return the metrics and timings
    return {
        'Dataset': dataset_name,
        'ARI': ari,
        'NMI': nmi,
        'AMI': ami,
        'Homogeneity': homogeneity,
        'Completeness': completeness,
        'V-score': v_score,
        'Fit Time': fit_time,
        'Predict Time': predict_time,
        'Total Time': total_time,
        'RAM Usage (GB)': ram_usage
    }

# Example usage
if __name__ == "__main__":
    # Run the clustering experiment with different metrics and gamma values
    results = []
    n_clusters = 3  # Set the number of clusters
    for metric in ['euclidean', 'dtw', 'softdtw']:
        for gamma in [None, 0.1, 1, 10]:
            if metric == 'softdtw' and gamma is None:
                continue  # Skip this combination

            # Load all dataset names
            datasets = UCR_UEA_datasets().list_datasets()

            # Loop over all datasets
            for dataset_name in datasets:
                result = run_clustering_experiment(dataset_name, n_clusters, metric, gamma)
                result['Metric'] = metric
                result['Gamma'] = gamma
                results.append(result)

    # Write the results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv('clustering_experiment_results.csv', index=False)