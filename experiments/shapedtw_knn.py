import signal

# Define a timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Experiment exceeded the time limit of 12 hours")

# Set the signal handler and a 12-hour alarm
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(12 * 60 * 60)  # 12 hours in seconds

try:
    try:
        import time
        import numpy as np
        import pandas as pd
        import logging
        import argparse
        import os
        import psutil
        import threading
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sktime.classification.distance_based import ShapeDTW
        from sktime.datasets import load_UCR_UEA_dataset, tsc_dataset_names
        from filelock import FileLock
    except Exception as e:
        print(f"An error occurred while importing modules: {e}")
        logger = None

    # Setup a basic logging configuration.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create a global lock
    lock = threading.Lock()

    def setup_logger(dataset, shape_function, subsequence_length, n_neighbors):
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f'shapedtw_{dataset}_{shape_function}_{subsequence_length}_{n_neighbors}.log')

        logger = logging.getLogger(f'shapedtw_{dataset}_{shape_function}_{subsequence_length}_{n_neighbors}')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


    def load_dataset_with_retry(dataset_name, logger, retries=3, delay=5):
        cache_dir = "cached_datasets"

        # Check if cached dataset exists
        if all(
                os.path.exists(os.path.join(cache_dir, f"{dataset_name}_{split}.npy"))
                for split in ["X_train", "y_train", "X_test", "y_test"]
        ):
            logger.info(f"Loading dataset '{dataset_name}' from cache.")
            X_train = np.load(os.path.join(cache_dir, f"{dataset_name}_X_train.npy"), allow_pickle=True)
            y_train = np.load(os.path.join(cache_dir, f"{dataset_name}_y_train.npy"), allow_pickle=True)
            X_test = np.load(os.path.join(cache_dir, f"{dataset_name}_X_test.npy"), allow_pickle=True)
            y_test = np.load(os.path.join(cache_dir, f"{dataset_name}_y_test.npy"), allow_pickle=True)
            return X_train, y_train, X_test, y_test

        # Retry downloading the dataset
        for attempt in range(retries):
            try:
                logger.info(f"Downloading dataset '{dataset_name}' (attempt {attempt + 1})...")
                X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
                X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

                # Cache the dataset locally
                np.save(os.path.join(cache_dir, f"{dataset_name}_X_train.npy"), X_train)
                np.save(os.path.join(cache_dir, f"{dataset_name}_y_train.npy"), y_train)
                np.save(os.path.join(cache_dir, f"{dataset_name}_X_test.npy"), X_test)
                np.save(os.path.join(cache_dir, f"{dataset_name}_y_test.npy"), y_test)

                return X_train, y_train, X_test, y_test
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} to load dataset '{dataset_name}' failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise


    def run_shapedtw_experiment(dataset_name, shape_function, subsequence_length, n_neighbors):
        result = {}
        try:
            logger = setup_logger(dataset_name, shape_function, subsequence_length, n_neighbors)
            X_train, y_train, X_test, y_test = load_dataset_with_retry(dataset_name, logger)

            # # Replace NaN values with 0
            # X_train = np.nan_to_num(X_train)
            # X_test = np.nan_to_num(X_test)

            clf = ShapeDTW(
                n_neighbors=n_neighbors,
                subsequence_length=subsequence_length,
                shape_descriptor_function=shape_function
            )

            process = psutil.Process(os.getpid())
            initial_ram_usage = process.memory_info().rss

            start_time = time.time()
            start_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

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
                'Experiment': 'ShapeDTW',
                'Dataset': dataset_name,
                'Shape Function': shape_function,
                'Subsequence Length': subsequence_length,
                'N Neighbors': n_neighbors,
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
            logger.error(f'Experiment exceeded the time limit: {e}', exc_info=True)
        except Exception as e:
            result.update({
                'Experiment Succeeded': False,
                'Comment': str(e)
            })
            logger.error(f'An error occurred: {e}', exc_info=True)

        df = pd.DataFrame(result, index=[0])

        # Define the directory and file path
        csv_dir = 'results'
        csv_file_path = os.path.join(csv_dir, 'shapedtw_experiment_results.csv')

        # Create the directory if it doesn't exist
        os.makedirs(csv_dir, exist_ok=True)

        lock_path = csv_file_path + '.lock'

        with FileLock(lock_path):
            if os.path.exists(csv_file_path):
                df.to_csv(csv_file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_file_path, mode='w', header=True, index=False)

        return result

    def main():
        parser = argparse.ArgumentParser(description='Run ShapeDTW experiment with specified parameters.')
        parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to use.')
        parser.add_argument('--shape_function', type=str, required=True, help='Shape descriptor function to use.')
        parser.add_argument('--subsequence_length', type=int, default=30, help='Subsequence length for ShapeDTW.')
        parser.add_argument('--n_neighbors', type=int, default=1, help='Number of neighbors for KNN in ShapeDTW.')
        args = parser.parse_args()

        for dataset in args.datasets:
            result = run_shapedtw_experiment(dataset, args.shape_function, args.subsequence_length, args.n_neighbors)
            print(result)

    if __name__ == "__main__":
        try:
            main()
        except TimeoutError as e:
            print(f"Experiment exceeded the time limit: {e}")
        except Exception as e:
            print(f"An error occurred while running the script: {e}")
        finally:
            signal.alarm(0)  # Disable the alarm

except Exception as e:
    print(f"An error occurred in the main script: {e}")
