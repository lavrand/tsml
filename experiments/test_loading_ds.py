from sktime.datasets import load_UCR_UEA_dataset

dataset_name = "ArrowHead"  # Replace with a valid dataset name
X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
