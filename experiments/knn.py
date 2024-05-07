import argparse
import os
import time
import pandas as pd
import psutil
import threading
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from tslearn.datasets import UCR_UEA_datasets
from tslearn.metrics import dtw, soft_dtw

# Create a global lock
lock = threading.Lock()

def setup_logger(dataset, k, metric, gamma):
    logger = logging.getLogger(f'knn_{dataset}_{k}_{metric}_{gamma}')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'knn_{dataset}_{k}_{metric}_{gamma}.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run_knn_experiment(dataset_name, k, metric, gamma):
    result = None

    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model = None

    if metric == 'euclidean':
        model = KNeighborsClassifier(n_neighbors=k)
    elif metric == 'dtw':
        model = KNeighborsClassifier(n_neighbors=k, metric=dtw)
    elif metric == 'softdtw' and gamma is not None:
        model = KNeighborsClassifier(n_neighbors=k, metric=lambda x, y: soft_dtw(x, y, gamma=gamma))

    if model is None:
        raise ValueError(f"Invalid metric: {metric} or gamma: {gamma}")

    process = psutil.Process(os.getpid())
    initial_ram_usage = process.memory_info().rss

    logger = setup_logger(dataset_name, k, metric, gamma)
    try:
        start_time = time.time()
        start_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        end_time = time.time()
        end_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        total_time = end_time - start_time

        final_ram_usage = process.memory_info().rss
        ram_usage = (final_ram_usage - initial_ram_usage) / (1024 ** 3)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        result = {
            'Experiment': 'KNN',
            'Dataset': dataset_name,
            'K': k,
            'Metric': metric,
            'Gamma': gamma,
            'Experiment Start Time': start_date_time,
            'Experiment End Time': end_date_time,
            'Experiment Total Time (seconds)': total_time,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'RAM Usage (GB)': ram_usage
        }

        df = pd.DataFrame(result, index=[0])
        csv_file_path = 'knn_experiment_results.csv'

        with lock:
            if os.path.exists(csv_file_path):
                df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_file_path, mode='w', header=True, index=False)

        logger.info('Experiment completed successfully')
    except Exception as e:
        logger.error(f'An error occurred: {e}', exc_info=True)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KNN experiment with specified parameters.')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to use.')
    parser.add_argument('--k', type=int, required=True, help='K value to use.')
    parser.add_argument('--metric', type=str, required=True, help='Distance metric to use.')
    parser.add_argument('--gamma', type=float, required=False, help='Gamma value for SoftDTW metric.')
    args = parser.parse_args()
    for dataset in args.datasets:
        result = run_knn_experiment(dataset, args.k, args.metric, args.gamma)
        print(result)