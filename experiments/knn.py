# Import necessary libraries
import itertools

from tslearn.datasets import UCR_UEA_datasets
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
from clearml import Task


# Function to run the KNN experiment
def run_knn_experiment(dataset_name, k_values, distance_metrics, gamma_values=None):
    # Initialize ClearML Task
    task = Task.init(project_name='Time Series Classification', task_name=f'KNN Experiment: {dataset_name}')
    task.connect_configuration(
        {"k_values": k_values, "distance_metrics": distance_metrics, "gamma_values": gamma_values})

    # Load dataset
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

    # Loop over different configurations
    for i, (k, metric) in enumerate(itertools.product(k_values, distance_metrics)):
        # Adjust for SoftDTW which needs a gamma parameter
        if metric == 'softdtw':
            if gamma_values is None:
                continue
            for gamma in gamma_values:
                model = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric=metric, metric_params={'gamma': gamma})
                model_name = f'{metric} (gamma={gamma})'
                run_and_log(model, model_name, X_train, y_train, X_test, y_test, task, i)
        else:
            model = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric=metric)
            model_name = metric
            run_and_log(model, model_name, X_train, y_train, X_test, y_test, task, i)

    task.close()


def run_and_log(model, model_name, X_train, y_train, X_test, y_test, task, iteration):
    start_time = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    end_time = time.time()

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')

    # Get the logger from the task
    logger = task.get_logger()

    # Ensure iteration is an integer
    iteration = int(iteration)  # Convert to int to avoid type issues

    # Log metrics and model info in ClearML
    logger.report_scalar('Accuracy', 'accuracy', accuracy, iteration)
    logger.report_scalar('F1 Score', 'f1', f1, iteration)
    logger.report_scalar('Precision', 'precision', precision, iteration)
    logger.report_scalar('Recall', 'recall',  recall, iteration)
    logger.report_scalar('Timing', 'run_time',  end_time - start_time, iteration)
    logger.flush()  # Ensure all metrics are uploaded


# Example usage
if __name__ == "__main__":
    dataset_name = "TwoPatterns"
    k_values = [1, 3, 5]
    distance_metrics = ['euclidean', 'dtw', 'softdtw']
    gamma_values = [0.1, 1, 10]

    run_knn_experiment(dataset_name, k_values, distance_metrics, gamma_values)
