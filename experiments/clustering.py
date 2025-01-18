try:
    try:
        import numpy as np
        import argparse
        import os
        import time
        import threading
        import logging
        import pandas as pd
        import psutil
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, \
            homogeneity_score, completeness_score, v_measure_score
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.datasets import UCR_UEA_datasets
        from tslearn.metrics import dtw, soft_dtw
        from filelock import FileLock
    except Exception as e:
        print(f"An error occurred while importing modules: {e}")
        logger = None

    # Create a global lock
    lock = threading.Lock()

    def setup_logger(dataset, n_clusters, metric, gamma):
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f'clustering_{dataset}_{n_clusters}_{metric}_{gamma}.log')

        logger = logging.getLogger(f'clustering_{dataset}_{n_clusters}_{metric}_{gamma}')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def run_clustering_experiment(dataset_name, n_clusters, metric='euclidean', gamma=None):
        result = {}
        try:
            X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)

            # Replace NaN values with 0
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

            # Determine the number of clusters based on the unique labels in y_train
            n_clusters = len(np.unique(y_train))
            model = None

            if metric == 'euclidean':
                model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric)
            elif metric == 'dtw':
                model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw')
            elif metric == 'softdtw':
                model = TimeSeriesKMeans(n_clusters=n_clusters, metric='softdtw')
            if model is None:
                raise ValueError(f"Invalid metric: {metric} or gamma: {gamma}")

            process = psutil.Process(os.getpid())
            initial_ram_usage = process.memory_info().rss

            logger = setup_logger(dataset_name, n_clusters, metric, gamma)
            start_fit = time.time()
            start_time = time.time()
            start_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

            model.fit(X_train)
            end_fit = time.time()

            start_predict_train = time.time()
            y_pred_train = model.predict(X_train)
            end_predict_train = time.time()

            start_predict_test = time.time()
            y_pred_test = model.predict(X_test)
            end_predict_test = time.time()

            end_time = time.time()
            end_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            total_time = end_time - start_time

            final_ram_usage = process.memory_info().rss
            ram_usage = (final_ram_usage - initial_ram_usage) / (1024 ** 3)

            result.update({
                'Experiment': 'Clustering',
                'Dataset': dataset_name,
                'Number of Clusters': n_clusters,
                'Metric': metric,
                'Gamma': gamma,
                'Experiment Start Time': start_date_time,
                'Experiment End Time': end_date_time,
                'Experiment Total Time (seconds)': total_time,
                'Adjusted Rand Index': adjusted_rand_score(y_test, y_pred_test),
                'Normalized Mutual Information': normalized_mutual_info_score(y_test, y_pred_test),
                'Adjusted Mutual Information': adjusted_mutual_info_score(y_test, y_pred_test),
                'Homogeneity': homogeneity_score(y_test, y_pred_test),
                'Completeness': completeness_score(y_test, y_pred_test),
                'V-score': v_measure_score(y_test, y_pred_test),
                'Fit Time': end_fit - start_fit,
                'Predict Time Train': end_predict_train - start_predict_train,
                'Predict Time Test': end_predict_test - start_predict_test,
                'Total Time': (end_fit - start_fit) + (end_predict_train - start_predict_train) + (
                        end_predict_test - start_predict_test),
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
            if logger:
                logger.error(f'An error occurred: {e}', exc_info=True)

        df = pd.DataFrame(result, index=[0])
        csv_file_path = 'clustering_experiment_results.csv'
        lock_path = csv_file_path + '.lock'

        with FileLock(lock_path):
            if os.path.exists(csv_file_path):
                df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_file_path, mode='w', header=True, index=False)

        return result

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Run clustering experiment with specified parameters.')
        parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to use.')
        parser.add_argument('--n_clusters', type=int, required=True, help='Number of clusters to use.')
        parser.add_argument('--metric', type=str, required=True, help='Distance metric to use.')
        parser.add_argument('--gamma', type=float, default=None, help='Gamma value for SoftDTW')
        args = parser.parse_args()
        for dataset in args.datasets:
            result = run_clustering_experiment(dataset, args.n_clusters, args.metric, args.gamma)
            print(result)
except Exception as e:
    print(f"An error occurred while running the script: {e}")