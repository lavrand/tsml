import signal

# Define a timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Experiment exceeded the time limit of 6 hours")

# Set the signal handler and a 6-hour alarm
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(6 * 60 * 60)  # 6 hours in seconds

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
        from tslearn.neighbors import KNeighborsTimeSeriesClassifier
        from tslearn.preprocessing import TimeSeriesResampler
        from filelock import FileLock
    except Exception as e:
        print(f"An error occurred while importing modules: {e}")
        logger = None

    # Create a global lock
    lock = threading.Lock()

    def setup_logger(dataset, k, metric, gamma):
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f'knn_{dataset}_{k}_{metric}_{gamma}.log')

        logger = logging.getLogger(f'knn_{dataset}_{k}_{metric}_{gamma}')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_dataset_with_retry(dataset_name, logger, retries=3, delay=5):
        ucr_uea = UCR_UEA_datasets(use_cache=True)
        for attempt in range(retries):
            try:
                X_train, y_train, X_test, y_test = ucr_uea.load_dataset(dataset_name)

                # Ensure consistent feature dimensions
                target_length = int(np.median([len(x) for x in X_train]))
                X_train, X_test = preprocess_features(X_train, X_test, target_length)

                return X_train, y_train, X_test, y_test
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} to load dataset {dataset_name} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise


    def preprocess_features(X_train, X_test, target_length):
        # Resample to ensure consistent length
        resampler = TimeSeriesResampler(sz=target_length)
        X_train = resampler.fit_transform(X_train)
        X_test = resampler.transform(X_test)

        # Replace NaNs with zeros
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

        return X_train, X_test

    def run_knn_experiment(dataset_name, k, metric, gamma):
        result = {}
        try:
            logger = setup_logger(dataset_name, k, metric, gamma)
            X_train, y_train, X_test, y_test = load_dataset_with_retry(dataset_name, logger)

            # Determine the desired length (e.g., the median length of the training set)
            lengths = [ts.shape[0] for ts in X_train]
            desired_length = int(np.median(lengths))

            # Preprocess features
            X_train, X_test = preprocess_features(X_train, X_test, desired_length)

            model = None

            if metric == 'euclidean':
                model = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric="euclidean")
            elif metric == 'dtw':
                model = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric="dtw")
            elif metric == 'softdtw' and gamma is not None:
                model = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric="softdtw", metric_params={"gamma": gamma})

            if model is None:
                raise ValueError(f"Invalid metric: {metric} or gamma: {gamma}")

            process = psutil.Process(os.getpid())
            initial_ram_usage = process.memory_info().rss

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

            logger.info('Test finished successfully')

        except TimeoutError as e:
            result.update({
                'Experiment Succeeded': False,
                'Comment': str(e)
            })
            if logger:
                logger.error(f'Experiment exceeded the time limit: {e}', exc_info=True)
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
        csv_file_path = os.path.join(csv_dir, 'knn_experiment_results.csv')

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
        parser = argparse.ArgumentParser(description='Run KNN experiment with specified parameters.')
        parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to use.')
        parser.add_argument('--k', type=int, required=True, help='K value to use.')
        parser.add_argument('--metric', type=str, required=True, help='Distance metric to use.')
        parser.add_argument('--gamma', type=float, default=None, help='Gamma value for SoftDTW')
        args = parser.parse_args()
        for dataset in args.datasets:
            result = run_knn_experiment(dataset, args.k, args.metric, args.gamma)
            print(result)
except TimeoutError as e:
    print(f"Experiment exceeded the time limit: {e}")
except Exception as e:
    print(f"An error occurred while running the script: {e}")
finally:
    signal.alarm(0)  # Disable the alarm