# Import necessary libraries
import argparse
import itertools
import multiprocessing
import os
import time
import traceback
import pandas as pd
import psutil
import threading
from clearml import Task
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tslearn.datasets import UCR_UEA_datasets
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

# Create a global lock
lock = threading.Lock()

def run_experiment(params):
    k, metric, gamma, X_train, y_train, X_test, y_test, task, i, dataset_name = params

    # Adjust for SoftDTW which needs a gamma parameter
    if metric == 'softdtw' and gamma is not None:
        model = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric=metric, metric_params={'gamma': gamma})
        model_name = f'{metric} (gamma={gamma})'
    else:
        model = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric=metric)
        model_name = metric

    # Run and log the model
    result = run_and_log(model, model_name, X_train, y_train, X_test, y_test, task, i, dataset_name)

    return result


def run_and_log(model, model_name, X_train, y_train, X_test, y_test, task, iteration, dataset_name):
    start_time = time.time()
    failed = False
    predictions = None  # Initialize predictions

    # Get initial RAM usage
    process = psutil.Process(os.getpid())
    initial_ram_usage = process.memory_info().rss  # in bytes

    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    except Exception as e:
        failed = True
        print(f"An error occurred: {traceback.format_exc()}")

    end_time = time.time()

    # Get final RAM usage
    final_ram_usage = process.memory_info().rss  # in bytes

    # Calculate RAM usage in GB
    ram_usage = (final_ram_usage - initial_ram_usage) / (1024 ** 3)

    # Calculate metrics
    if predictions is not None:
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
    else:
        accuracy = f1 = precision = recall = None

    # Get the logger from the task
    logger = task.get_logger()

    # Explicitly report the iteration to ClearML
    logger.report_scalar('Iteration', 'iteration', iteration, iteration)


    # Log metrics and model info in ClearML
    if accuracy is not None and f1 is not None and precision is not None and recall is not None:
        logger.report_scalar('Accuracy', 'accuracy', accuracy, iteration)
        logger.report_scalar('F1 Score', 'f1', f1, iteration)
        logger.report_scalar('Precision', 'precision', precision, iteration)
        logger.report_scalar('Recall', 'recall', recall, iteration)
    logger.report_scalar('Timing', 'run_time', end_time - start_time, iteration)
    logger.report_scalar('RAM Usage (GB)', 'ram_usage', ram_usage, iteration)
    logger.flush()  # Ensure all metrics are uploaded

    # Return the metrics and other information for the table
    return {
        'Dataset Source': 'UCR',
        'Dataset': dataset_name,
        'Train Size': len(X_train),
        'Test Size': len(X_test),
        'Length': len(X_train[0]),
        'Method Group': 'KNN',
        'Method': model_name,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Run Time': end_time - start_time,
        'Failed': failed,
        'RAM Usage (GB)': ram_usage
    }


def run_knn_experiment(k_values, distance_metrics, gamma_values=None):
    # Initialize ClearML Task
    task = Task.init(project_name='Time Series Classification', task_name='KNN Experiment')
    task.connect_configuration(
        {"k_values": k_values, "distance_metrics": distance_metrics, "gamma_values": gamma_values})

    # Load all dataset names
    datasets = UCR_UEA_datasets().list_datasets()

    # Prepare a list to store the results for all datasets
    results = []

    # Loop over all datasets
    for dataset_name in datasets:
        # Load dataset
        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

        # Generate combinations of k_values, distance_metrics, and gamma_values
        combinations = list(itertools.product(k_values, distance_metrics, gamma_values or [None]))

        # Prepare the parameters for each experiment
        experiment_params = [(k, metric, gamma, X_train, y_train, X_test, y_test, task, i, dataset_name) for i, (k, metric, gamma) in enumerate(combinations)]

        # Create a multiprocessing pool and run the experiments in parallel
        with multiprocessing.Pool() as pool:
            results.extend(pool.map(run_experiment, experiment_params))

        task.close()

    # Write the results to a CSV file
    df = pd.DataFrame(results)
    csv_file_path = 'knn_experiment_results.csv'

    # Use the lock to prevent race conditions
    with lock:
        # Check if the file exists
        if os.path.exists(csv_file_path):
            # If the file exists, append without the header
            df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            # If the file does not exist, create it with a header
            df.to_csv(csv_file_path, mode='w', header=True, index=False)

    # Upload the CSV file to ClearML
    task.upload_artifact('knn_experiment_results', csv_file_path)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Run KNN experiment with specified parameters.')
    parser.add_argument('--k_values', type=int, nargs='+', required=True, help='List of K values to use.')
    parser.add_argument('--distance_metrics', type=str, nargs='+', required=True, help='List of distance metrics to use.')
    parser.add_argument('--gamma_values', type=float, nargs='*', required=False, help='List of gamma values to use for SoftDTW metric.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the KNN experiment with the specified parameters
    run_knn_experiment(args.k_values, args.distance_metrics, args.gamma_values)