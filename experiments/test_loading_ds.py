from sktime.datasets import load_UCR_UEA_dataset

try:
    print("Testing Adiac dataset...")
    X_train, y_train = load_UCR_UEA_dataset("Adiac", split="train", return_X_y=True)
    X_test, y_test = load_UCR_UEA_dataset("Adiac", split="test", return_X_y=True)
    print("Adiac dataset loaded successfully!")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
except Exception as e:
    print(f"Failed to load Adiac dataset: {e}")
