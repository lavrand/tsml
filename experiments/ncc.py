try:
    try:
        import argparse
        import os
        import time
        import pandas as pd
        import psutil
        import threading
        import logging
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
        from tslearn.datasets import UCR_UEA_datasets
        from tslearn.metrics import dtw, soft_dtw
        from filelock import FileLock
    except Exception as e:
        print(f"An error occurred while importing modules: {e}")
        logger = None

    # Create a global lock
    lock = threading.Lock()

    def setup_logger(dataset, metric, gamma):
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f'ncc_{dataset}_{metric}_{gamma}.log')

        logger = logging.getLogger(f'ncc_{dataset}_{metric}_{gamma}')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def run_ncc_experiment(dataset_name, metric='euclidean', gamma=None):
        result = {}
        try:
            X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

            # Replace NaN values with 0
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            model = None

            if metric == 'euclidean':
                model = NearestCentroid()
            elif metric == 'dtw':
                model = KNeighborsClassifier(n_neighbors=1, metric=dtw)
            elif metric == 'softdtw' and gamma is not None:
                model = KNeighborsClassifier(n_neighbors=1, metric=lambda x, y: soft_dtw(x, y, gamma=gamma))

            if model is None:
                raise ValueError(f"Invalid metric: {metric} or gamma: {gamma}")

            process = psutil.Process(os.getpid())
            initial_ram_usage = process.memory_info().rss

            logger = setup_logger(dataset_name, metric, gamma)
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

            result.update({
                'Experiment': 'NCC',
                'Dataset': dataset_name,
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
            })

            result.update({
                'Experiment Succeeded': True,
                'Comment': 'Experiment completed successfully'
            })

            logger.info('Test finished successfully')

        except Exception as e:
            result.update({
                'Experiment Succeeded': False,
                'Comment': str(e)
            })
            if logger:
                logger.error(f'An error occurred: {e}', exc_info=True)

        df = pd.DataFrame(result, index=[0])

        # Define the directory and file path
        csv_dir = 'results'
        csv_file_path = os.path.join(csv_dir, 'ncc_experiment_results.csv')

        # Create the directory if it doesn't exist
        os.makedirs(csv_dir, exist_ok=True)

        lock_path = csv_file_path + '.lock'

        with FileLock(lock_path):
            if os.path.exists(csv_file_path):
                df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_file_path, mode='w', header=True, index=False)

        return result

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Run NCC experiment with specified parameters.')
        parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to use.')
        parser.add_argument('--metric', type=str, required=True, help='Distance metric to use.')
        parser.add_argument('--gamma', type=float, default=None, help='Gamma value for SoftDTW')
        args = parser.parse_args()
        for dataset in args.datasets:
            result = run_ncc_experiment(dataset, args.metric, args.gamma)
            print(result)
except Exception as e:
    print(f"An error occurred while running the script: {e}")