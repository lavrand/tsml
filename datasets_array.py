import csv

# Read the dataset names from the CSV file
with open('datasets.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header
    datasets = [row[0] for row in reader]

# Sort the dataset names alphabetically
datasets.sort()

# Print the dataset names in the desired format
print("datasets = [")
for i, dataset in enumerate(datasets):
    if i == len(datasets) - 1:
        print(f"    '{dataset}'")
    else:
        print(f"    '{dataset}',")
print("]")