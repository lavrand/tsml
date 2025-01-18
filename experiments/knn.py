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
        from tslearn.datasets import UCR_UEA_datasets
        from tslearn.metrics import dtw, soft_dtw
        from sklearn.neighbors import KNeighborsClassifier
        from filelock import FileLock
    except Exception as e:
        print(f"An error occurred while importing modules: {e}")

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
            })

            result.update({
                'Experiment Succeeded': True,
                'Comment': 'Experiment completed successfully'
            })

        except Exception as e:
            result.update({
                'Experiment Succeeded': False,
                'Comment': str(e)
            })
            logger.error(f'An error occurred: {e}', exc_info=True)

        df = pd.DataFrame(result, index=[0])
        csv_file_path = 'knn_experiment_results.csv'
        lock_path = csv_file_path + '.lock'

        with FileLock(lock_path):
            if os.path.exists(csv_file_path):
                df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_file_path, mode='w', header=True, index=False)

        return result

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Run KNN experiment with specified parameters.')
        parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to use.')
        parser.add_argument('--k', type=int, required=True, help='K value to use.')
        parser.add_argument('--metric', type=str, required=True, help='Distance metric to use.')
        parser.add_argument('--gamma', type=float, default=None, help='Gamma value for SoftDTW')
        args = parser.parse_args()
        for dataset in args.datasets:
            result = run_knn_experiment(dataset, args.k, args.metric, args.gamma)
            print(result)
except Exception as e:
    print(f"An error occurred while running the script: {e}")