import os
import time
import numpy as np
from sktime.datasets import load_UCR_UEA_dataset

# Define the directory to store cached datasets
CACHE_DIR = "cached_datasets"
os.makedirs(CACHE_DIR, exist_ok=True)

# List of datasets to pre-download
DATASETS = (
    "ACSF1", "ArrowHead", "ItalyPowerDemand", "GunPoint", "OSULeaf",
    "Adiac", "Beef", "BeetleFly", "BirdChicken", "BME", "Car",
    "CBF", "Chinatown", "ChlorineConcentration", "CinCECGTorso",
    "Coffee", "Computers", "CricketX", "CricketY", "CricketZ",
    "Crop", "DiatomSizeReduction", "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "Earthquakes",
    "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices",
    "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel",
    "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish",
    "FordA", "FordB", "FreezerRegularTrain", "FreezerSmallTrain",
    "Fungi", "GunPointAgeSpan", "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung", "Ham", "HandOutlines", "Haptics",
    "Herring", "HouseTwenty", "InlineSkate", "InsectEPGRegularTrain",
    "InsectEPGSmallTrain", "InsectWingbeatSound", "LargeKitchenAppliances",
    "Lightning2", "Lightning7", "Mallat", "Meat", "MedicalImages",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW", "MixedShapesRegularTrain", "MixedShapesSmallTrain",
    "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2",
    "OliveOil", "PhalangesOutlinesCorrect", "Phoneme", "PigAirwayPressure",
    "PigArtPressure", "PigCVP", "Plane", "PowerCons",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW", "RefrigerationDevices", "Rock", "ScreenType",
    "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2",
    "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SmoothSubspace",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarlightCurves",
    "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl",
    "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG",
    "TwoPatterns", "UMD", "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine",
    "WordSynonyms", "Worms", "WormsTwoClass", "Yoga",
)

# Create a lock file for serialization
LOCK_FILE = os.path.join(CACHE_DIR, "cache_lock.lock")


def cache_dataset(dataset_name, retries=5, initial_delay=5):
    """
    Cache a dataset with retries and adaptive delays to avoid cluster-specific issues.

    Parameters:
    - dataset_name (str): The name of the dataset to cache.
    - retries (int): Number of retry attempts.
    - initial_delay (int): Initial delay in seconds between retries.
    """
    for attempt in range(1, retries + 1):
        try:
            # Acquire lock to serialize dataset loading
            with open(LOCK_FILE, "w") as lock:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Acquiring lock for {dataset_name}")
                lock.write(f"Locking dataset: {dataset_name}\n")

                print(f"Attempting to cache dataset: {dataset_name} (Attempt {attempt})")

                # Load the dataset
                X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
                X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

                # Save datasets to cache
                np.save(os.path.join(CACHE_DIR, f"{dataset_name}_X_train.npy"), X_train)
                np.save(os.path.join(CACHE_DIR, f"{dataset_name}_y_train.npy"), y_train)
                np.save(os.path.join(CACHE_DIR, f"{dataset_name}_X_test.npy"), X_test)
                np.save(os.path.join(CACHE_DIR, f"{dataset_name}_y_test.npy"), y_test)

                print(f"Successfully cached dataset: {dataset_name}")
                return  # Exit the loop on success
        except Exception as e:
            print(f"Failed to cache dataset {dataset_name} on attempt {attempt}: {e}")
            if attempt < retries:
                delay = initial_delay * (2 ** (attempt - 1))  # Exponential backoff
                print(f"Retrying {dataset_name} in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Exhausted retries for dataset {dataset_name}. Moving on.")
        finally:
            # Release lock
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)


if __name__ == "__main__":
    # Process datasets sequentially with delays
    for dataset in DATASETS:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing dataset: {dataset}")
        cache_dataset(dataset, retries=5, initial_delay=20)
        time.sleep(40)  # Fixed delay between datasets to avoid overlapping
