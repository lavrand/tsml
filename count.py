import csv

datasets = [
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'DodgerLoopDay',
    'DodgerLoopGame',
    'DodgerLoopWeekend',
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'MelbournePedestrian',
    'PLAID',
    'PickupGestureWiimoteZ',
    'ShakeGestureWiimoteZ',
    'StarlightCurves'
]
print(f"Number of datasets: {len(datasets)}")

with open('datasets.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Dataset'])
    for dataset in datasets:
        writer.writerow([dataset])