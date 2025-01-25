import csv

datasets = [
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'DodgerLoopDay',
    'DodgerLoopGame',
    'DodgerLoopWeekend',
    'MelbournePedestrian',
    'PLAID',
    'StarLightCurves'
]
print(f"Number of datasets: {len(datasets)}")

with open('datasets.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Dataset'])
    for dataset in datasets:
        writer.writerow([dataset])