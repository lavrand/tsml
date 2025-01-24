import os
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe

# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the 'ds' directory
DATASET_DIR = os.path.join(current_dir, 'ds')

# Path to the cached datasets directory
CACHE_DIR = os.path.join(current_dir, 'cached_datasets')
os.makedirs(CACHE_DIR, exist_ok=True)

# List of datasets to process
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

def cache_dataset(dataset_name):
    """
    Cache the specified dataset by reading the .ts file and converting it to NumPy format.
    """
    try:
        print(f"Processing dataset: {dataset_name}")

        # Construct file paths for train and test sets
        train_file = os.path.join(DATASET_DIR, dataset_name, f"{dataset_name}_TRAIN.ts")
        test_file = os.path.join(DATASET_DIR, dataset_name, f"{dataset_name}_TEST.ts")

        # Check if files exist
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Train file not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        # Load train and test datasets
        X_train, y_train = load_from_tsfile_to_dataframe(train_file)
        X_test, y_test = load_from_tsfile_to_dataframe(test_file)

        # Log the structure of X_train and X_test
        # print(f"Type of X_train: {type(X_train)}")
        # print(f"First 5 rows of X_train: {X_train.head()}")
        # print(f"Type of first element in X_train: {type(X_train.iloc[0, 0])}")
        #
        # print(f"Type of X_test: {type(X_test)}")
        # print(f"First 5 rows of X_test: {X_test.head()}")
        # print(f"Type of first element in X_test: {type(X_test.iloc[0, 0])}")

        # Convert the dataframes to numpy arrays
        X_train_np = np.array([row[0] for row in X_train.itertuples(index=False)])
        X_test_np = np.array([row[0] for row in X_test.itertuples(index=False)])

        # Save the datasets to the cache directory
        np.save(os.path.join(CACHE_DIR, f"{dataset_name}_X_train.npy"), X_train_np)
        np.save(os.path.join(CACHE_DIR, f"{dataset_name}_y_train.npy"), np.array(y_train))
        np.save(os.path.join(CACHE_DIR, f"{dataset_name}_X_test.npy"), X_test_np)
        np.save(os.path.join(CACHE_DIR, f"{dataset_name}_y_test.npy"), np.array(y_test))

        print(f"Successfully cached dataset: {dataset_name}")
    except Exception as e:
        print(f"Failed to cache dataset {dataset_name}: {e}")

if __name__ == "__main__":
    # Process each dataset in the list
    for dataset in DATASETS:
        cache_dataset(dataset)
