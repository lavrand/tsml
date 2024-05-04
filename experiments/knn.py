# Import necessary libraries
import itertools
import multiprocessing
import os
import time
import traceback
import pandas as pd
import psutil
from clearml import Task
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tslearn.datasets import UCR_UEA_datasets
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

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
    ram_usages = []
    predictions = None  # Initialize predictions

    try:
        model.fit(X_train, y_train)

        # Calculate RAM usage periodically during model training
        while True:
            process = psutil.Process(os.getpid())
            ram_usages.append(process.memory_info().rss)  # in bytes
            time.sleep(1)  # pause for 1 second
            if model.training_finished:  # assuming the model has a property that indicates if training is finished
                break

        predictions = model.predict(X_test)

    except Exception as e:
        failed = True
        print(f"An error occurred: {traceback.format_exc()}")

    end_time = time.time()

    # Calculate metrics
    if predictions is not None:
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
    else:
        accuracy = f1 = precision = recall = None

    # Calculate average and maximum RAM usage
    ram_average = sum(ram_usages) / len(ram_usages) if ram_usages else None
    ram_maximum = max(ram_usages) if ram_usages else None

    # Get the logger from the task
    logger = task.get_logger()

    # Log metrics and model info in ClearML
    if accuracy is not None and f1 is not None and precision is not None and recall is not None:
        logger.report_scalar('Accuracy', 'accuracy', accuracy, iteration)
        logger.report_scalar('F1 Score', 'f1', f1, iteration)
        logger.report_scalar('Precision', 'precision', precision, iteration)
        logger.report_scalar('Recall', 'recall', recall, iteration)
    logger.report_scalar('Timing', 'run_time', end_time - start_time, iteration)
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
        'RAM Average': ram_average,
        'RAM Maximum': ram_maximum
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
    csv_file_path = 'experiment_results.csv'
    df.to_csv(csv_file_path, index=False)

    # Upload the CSV file to ClearML
    task.upload_artifact('experiment_results', csv_file_path)

# Example usage
if __name__ == "__main__":
    k_values = [1, 3, 5]
    distance_metrics = ['euclidean', 'dtw', 'softdtw']
    gamma_values = [0.1, 1, 10]

    run_knn_experiment(k_values, distance_metrics, gamma_values)