import argparse
import os
import time
import threading
import logging
import pandas as pd
import psutil
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, \
    homogeneity_score, completeness_score, v_measure_score, accuracy_score, f1_score, precision_score, recall_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import UCR_UEA_datasets
from tslearn.metrics import dtw, soft_dtw

# Create a global lock
lock = threading.Lock()

def setup_logger(dataset, n_clusters, metric, gamma):
    logger = logging.getLogger(f'clustering_{dataset}_{n_clusters}_{metric}_{gamma}')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'clustering_{dataset}_{n_clusters}_{metric}_{gamma}.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run_clustering_experiment(dataset_name, n_clusters, metric='euclidean', gamma=None):
    result = None
    try:
        from clearml import Task
    except ImportError:
        print("clearml is not installed on this system.")
        return

    task = Task.init(project_name='Time Series Classification', task_name='Clustering Experiment')
    task.connect_configuration(
        {"dataset_name": dataset_name, "n_clusters": n_clusters, "metric": metric, "gamma": gamma})

    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

    model = None

    if metric == 'euclidean':
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric)
    elif metric == 'dtw':
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric=dtw)
    elif metric == 'softdtw' and gamma is not None:
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric=lambda x, y: soft_dtw(x, y, gamma=gamma))

    if model is None:
        raise ValueError(f"Invalid metric: {metric} or gamma: {gamma}")

    process = psutil.Process(os.getpid())
    initial_ram_usage = process.memory_info().rss

    logger = setup_logger(dataset_name, n_clusters, metric, gamma)
    try:
        start_fit = time.time()
        model.fit(X_train)
        end_fit = time.time()

        start_predict_train = time.time()
        y_pred_train = model.predict(X_train)
        end_predict_train = time.time()

        start_predict_test = time.time()
        y_pred_test = model.predict(X_test)
        end_predict_test = time.time()

        final_ram_usage = process.memory_info().rss
        ram_usage = (final_ram_usage - initial_ram_usage) / (1024 ** 3)

        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='macro')
        precision = precision_score(y_test, y_pred_test, average='macro')
        recall = recall_score(y_test, y_pred_test, average='macro')

        fit_time = end_fit - start_fit
        predict_time_train = end_predict_train - start_predict_train
        predict_time_test = end_predict_test - start_predict_test
        total_time = fit_time + predict_time_train + predict_time_test

        result = {
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Fit Time': fit_time,
            'Predict Time Train': predict_time_train,
            'Predict Time Test': predict_time_test,
            'Total Time': total_time,
            'RAM Usage (GB)': ram_usage
        }

        df = pd.DataFrame(result, index=[0])
        csv_file_path = 'clustering_experiment_results.csv'

        with lock:
            if os.path.exists(csv_file_path):
                df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_file_path, mode='w', header=True, index=False)

        task.upload_artifact('clustering_experiment_results', csv_file_path)
        logger.info('Experiment completed successfully')
    except Exception as e:
        logger.error(f'An error occurred: {e}', exc_info=True)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run clustering experiment with specified parameters.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use.')
    parser.add_argument('--n_clusters', type=int, required=True, help='Number of clusters to use.')
    parser.add_argument('--metric', type=str, required=True, help='Distance metric to use.')
    parser.add_argument('--gamma', type=float, required=False, help='Gamma value for SoftDTW metric.')
    args = parser.parse_args()
    result = run_clustering_experiment(args.dataset, args.n_clusters, args.metric, args.gamma)
    print(result)