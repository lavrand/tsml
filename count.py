import csv

datasets = [
    'Crop',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'EthanolLevel',
    'FreezerRegularTrain',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'HouseTwenty',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'MixedShapesSmallTrain',
    'PLAID',
    'PickupGestureWiimoteZ',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'PowerCons',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'UMD'
]
print(f"Number of datasets: {len(datasets)}")

with open('datasets.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Dataset'])
    for dataset in datasets:
        writer.writerow([dataset])