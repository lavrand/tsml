import argparse
import os
import time
import pandas as pd
import psutil
import threading
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestCentroid
from tslearn.datasets import UCR_UEA_datasets
from tslearn.metrics import dtw, soft_dtw

# Create a global lock
lock = threading.Lock()

def setup_logger(dataset, metric, gamma):
    logger = logging.getLogger(f'ncc_{dataset}_{metric}_{gamma}')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'ncc_{dataset}_{metric}_{gamma}.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run_ncc_experiment(dataset_name, metric='euclidean', gamma=None):
    try:
        from clearml import Task
    except ImportError:
        print("clearml is not installed on this system.")
        return

    task = Task.init(project_name='Time Series Classification', task_name='NCC Experiment')
    task.connect_configuration(
        {"dataset_name": dataset_name, "metric": metric, "gamma": gamma})

    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

    model = None

    if metric == 'euclidean':
        model = NearestCentroid()
    elif metric == 'dtw':
        model = NearestCentroid(metric=dtw)
    elif metric == 'softdtw' and gamma is not None:
        model = NearestCentroid(metric=lambda x, y: soft_dtw(x, y, gamma=gamma))

    if model is None:
        raise ValueError(f"Invalid metric: {metric} or gamma: {gamma}")

    process = psutil.Process(os.getpid())
    initial_ram_usage = process.memory_info().rss

    logger = setup_logger(dataset_name, metric, gamma)
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        final_ram_usage = process.memory_info().rss
        ram_usage = (final_ram_usage - initial_ram_usage) / (1024 ** 3)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        result = {
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'RAM Usage (GB)': ram_usage
        }

        df = pd.DataFrame(result, index=[0])
        csv_file_path = 'ncc_experiment_results.csv'

        with lock:
            if os.path.exists(csv_file_path):
                df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_file_path, mode='w', header=True, index=False)

        task.upload_artifact('ncc_experiment_results', csv_file_path)
        logger.info('Experiment completed successfully')
    except Exception as e:
        logger.error(f'An error occurred: {e}', exc_info=True)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NCC experiment with specified parameters.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use.')
    parser.add_argument('--metric', type=str, required=True, help='Distance metric to use.')
    parser.add_argument('--gamma', type=float, required=False, help='Gamma value for SoftDTW metric.')
    args = parser.parse_args()
    result = run_ncc_experiment(args.dataset, args.metric, args.gamma)
    print(result)