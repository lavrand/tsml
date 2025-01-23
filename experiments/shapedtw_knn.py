import time
import numpy as np
import pandas as pd
import logging

from sktime.classification.distance_based import ShapeDTW
from sktime.datasets import load_UCR_UEA_dataset
from sktime.datasets import tsc_dataset_names

# Setup a basic logging configuration. 
# You can configure this further (file logs, log levels, etc.) in your main script.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_shapedtw_on_dataset(
    dataset_name: str,
    shape_function: str,
    subsequence_length: int = 30,
    n_neighbors: int = 1
) -> dict:
    """
    Runs ShapeDTW on a single dataset with a given shape descriptor,
    logs relevant information, and returns a dictionary of results.

    Parameters
    ----------
    dataset_name : str
        Name of the UCR dataset.
    shape_function : str
        The shape descriptor function to use, e.g., "raw", "hog1d", "derivative", etc.
    subsequence_length : int, optional
        The subsequence length for ShapeDTW, by default 30.
    n_neighbors : int, optional
        Number of neighbors for the underlying KNN in ShapeDTW, by default 1.

    Returns
    -------
    dict
        Dictionary containing dataset name, shape function, accuracy, runtime, 
        size of train/test sets, etc.
    """
    # Load train/test data
    X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train")
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test")

    # Create the ShapeDTW classifier
    clf = ShapeDTW(
        n_neighbors=n_neighbors,
        subsequence_length=subsequence_length,
        shape_descriptor_function=shape_function
        # Additional ShapeDTW params can go here
    )

    # Fit and predict
    start_time = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end_time = time.time()

    # Compute accuracy and timing
    accuracy = np.mean(y_pred == y_test)
    run_time = end_time - start_time

    # Log progress/info
    logger.info(
        "Finished: dataset=%s, shape_function=%s, accuracy=%.4f, runtime=%.2f seconds, "
        "n_train=%d, n_test=%d",
        dataset_name,
        shape_function,
        accuracy,
        run_time,
        len(y_train),
        len(y_test),
    )

    # Return results as a dictionary
    return {
        "dataset": dataset_name,
        "shape_function": shape_function,
        "accuracy": accuracy,
        "run_time_sec": run_time,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "subsequence_length": subsequence_length,
        "n_neighbors": n_neighbors
    }


def run_shapedtw_experiments(
    dataset_list=None,
    shape_functions=None,
    subsequence_length: int = 30,
    n_neighbors: int = 1
) -> pd.DataFrame:
    """
    Runs ShapeDTW on multiple datasets and shape descriptor functions,
    returning the results as a pandas DataFrame.

    Parameters
    ----------
    dataset_list : list of str, optional
        List of dataset names. If None, defaults to univariate_equal_length from sktime.
    shape_functions : list of str, optional
        List of shape descriptor functions to test. If None, defaults to ["raw", "hog1d"].
    subsequence_length : int, optional
        Subsequence length for ShapeDTW, by default 30.
    n_neighbors : int, optional
        Number of neighbors for ShapeDTW, by default 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        [dataset, shape_function, accuracy, run_time_sec, n_train, n_test, 
         subsequence_length, n_neighbors].
    """
    if dataset_list is None:
        dataset_list = tsc_dataset_names.univariate_equal_length
    if shape_functions is None:
        shape_functions = ["raw", "hog1d"]

    results = []
    for dataset_name in dataset_list:
        for shape_func in shape_functions:
            result_dict = run_shapedtw_on_dataset(
                dataset_name=dataset_name,
                shape_function=shape_func,
                subsequence_length=subsequence_length,
                n_neighbors=n_neighbors
            )
            results.append(result_dict)

    return pd.DataFrame(results)


def main():
    # Example usage:
    # Choose a dataset or a list of datasets
    example_dataset = "ECG200"

    # Choose shape descriptor functions
    shape_funcs = ["raw", "hog1d"]

    # Run the experiment (single or multiple datasets)
    df_results = run_shapedtw_experiments(
        dataset_list=[example_dataset],
        shape_functions=shape_funcs,
        subsequence_length=30,
        n_neighbors=1
    )

    # Print out the results DataFrame
    print(df_results)


# If you run this script directly: 
if __name__ == "__main__":
    main()

# 4 CPU, 16 RAM
# pip install setuptools==68.0.0  # A version compatible with Python 3.12
# pip install distlib