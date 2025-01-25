import os
import numpy as np

# DTW_N_PARALLEL_DATASETS = 8
DTW_N_PARALLEL_DATASETS = 128
KNN_N_PARALLEL_DATASETS = 17

IS_RUN_DTW = True
IS_RUN_KNN = True

# dtw "raw", "hog1d"
# dtw_datasets = [
#     'AllGestureWiimoteY',
#     'AllGestureWiimoteZ',
#     'DodgerLoopDay',
#     'DodgerLoopGame',
#     'DodgerLoopWeekend',
#     'MelbournePedestrian',
#     'PLAID',
#     'StarLightCurves'
# ]

# dtw "derivative"
dtw_datasets = [
    'ACSF1',
    'Adiac',
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'ArrowHead',
    'BME',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'CBF',
    'Car',
    'Chinatown',
    'ChlorineConcentration',
    'CinCECGTorso',
    'Coffee',
    'Computers',
    'CricketX',
    'CricketY',
    'CricketZ',
    'Crop',
    'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW',
    'DodgerLoopDay',
    'DodgerLoopGame',
    'DodgerLoopWeekend',
    'ECG200',
    'ECG5000',
    'ECGFiveDays',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'Earthquakes',
    'ElectricDevices',
    'EthanolLevel',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    'FiftyWords',
    'Fish',
    'FordA',
    'FordB',
    'FreezerRegularTrain',
    'FreezerSmallTrain',
    'Fungi',
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GunPoint',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'Ham',
    'HandOutlines',
    'Haptics',
    'Herring',
    'HouseTwenty',
    'InlineSkate',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'InsectWingbeatSound',
    'ItalyPowerDemand',
    'LargeKitchenAppliances',
    'Lightning2',
    'Lightning7',
    'Mallat',
    'Meat',
    'MedicalImages',
    'MelbournePedestrian',
    'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW',
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'MoteStrain',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    'OSULeaf',
    'OliveOil',
    'PLAID',
    'PhalangesOutlinesCorrect',
    'Phoneme',
    'PickupGestureWiimoteZ',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'Plane',
    'PowerCons',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect',
    'ProximalPhalanxTW',
    'RefrigerationDevices',
    'Rock',
    'ScreenType',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'ShapeletSim',
    'ShapesAll',
    'SmallKitchenAppliances',
    'SmoothSubspace',
    'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2',
    'StarLightCurves',
    'Strawberry',
    'SwedishLeaf',
    'Symbols',
    'SyntheticControl',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    'TwoLeadECG',
    'TwoPatterns',
    'UMD',
    'UWaveGestureLibraryAll',
    'UWaveGestureLibraryX',
    'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ',
    'Wafer',
    'Wine',
    'WordSynonyms',
    'Worms',
    'WormsTwoClass',
    'Yoga'
]

# knn
knn_datasets = [
    'Crop',
    'EOGVerticalSignal',
    'EthanolLevel',
    'FordA',
    'HandOutlines',
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    'PLAID',
    'PickupGestureWiimoteZ',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'StarLightCurves',
    'UWaveGestureLibraryAll'
]

# Split the datasets into N_PARALLEL_DATASETS groups
dtw_dataset_groups = np.array_split(dtw_datasets, DTW_N_PARALLEL_DATASETS)
knn_dataset_groups = np.array_split(knn_datasets, KNN_N_PARALLEL_DATASETS)

k_values = [1, 3, 5]
distance_metrics = ['euclidean', 'dtw', 'softdtw']
gamma_values = [0.1, 1, 10]
n_clusters = [3]

sbatch_template_shapedtw_knn = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-23:50:00
#SBATCH --job-name shapedtw_knn_{shape_function}
#SBATCH --output /cs_storage/andreyl/dtw_logs/dtw-%J.out  # Output log file (%J will be replaced by the job ID)
#SBATCH --error /cs_storage/andreyl/dtw_error_logs/dtw-%J.err   # Error log file (%J will be replaced by the job ID)
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=FAIL
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --tasks=1

module load anaconda
source activate tsml

bash shapedtw_knn.sh {datasets} {shape_function}
"""

# shape_functions = ["raw", "hog1d"]
shape_functions = ["derivative"]

for group in dtw_dataset_groups:
    if len(group) == 0:
        continue
    group = list(group)  # Convert array back to list
    first_dataset = group[0]
    last_dataset = group[-1]
    for shape_function in shape_functions:
        sbatch_content_shapedtw_knn = sbatch_template_shapedtw_knn.format(datasets="_".join(group), shape_function=shape_function)
        with open(f"shapedtw_knn_{first_dataset}_{last_dataset}_{shape_function}.sbatch", "w") as f:
            f.write(sbatch_content_shapedtw_knn)

sbatch_template_knn = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-23:50:00
#SBATCH --job-name knn_{k}_{metric}_{gamma}
#SBATCH --output /cs_storage/andreyl/pancake/knn_{k}_{metric}_{gamma}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=FAIL
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --tasks=1

module load anaconda
source activate tsml

bash run_knn.sh {datasets} {k} {metric} {gamma}
"""

sbatch_template_ncc = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-23:50:00
#SBATCH --job-name ncc_{metric}_{gamma}
#SBATCH --output /cs_storage/andreyl/pancake/ncc_{metric}_{gamma}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=FAIL
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --tasks=1

module load anaconda
source activate tsml

bash run_ncc.sh {datasets} {metric} {gamma}
"""

sbatch_template_clustering = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-23:50:00
#SBATCH --job-name clustering_{n_cluster}_{metric}_{gamma}
#SBATCH --output /cs_storage/andreyl/pancake/clustering_{n_cluster}_{metric}_{gamma}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=FAIL
#SBATCH --mem=20G
#SBATCH --cpus-per-task=10
#SBATCH --tasks=1

module load anaconda
source activate tsml

bash run_clustering.sh {datasets} {n_cluster} {metric} {gamma}
"""

for group in knn_dataset_groups:
    if len(group) == 0:
        continue
    group = list(group)  # Convert array back to list
    first_dataset = group[0]
    last_dataset = group[-1]
    for k in k_values:
        for metric in distance_metrics:
            if metric == 'softdtw':
                for gamma in gamma_values:
                    sbatch_content_knn = sbatch_template_knn.format(datasets="_".join(group), k=k, metric=metric, gamma=gamma)
                    with open(f"knn_{first_dataset}_{last_dataset}_{k}_{metric}_{gamma}.sbatch", "w") as f:
                        f.write(sbatch_content_knn)
            else:
                sbatch_content_knn = sbatch_template_knn.format(datasets="_".join(group), k=k, metric=metric, gamma="")
                with open(f"knn_{first_dataset}_{last_dataset}_{k}_{metric}.sbatch", "w") as f:
                    f.write(sbatch_content_knn)

    for metric in distance_metrics:
        if metric == 'softdtw':
            for gamma in gamma_values:
                sbatch_content_ncc = sbatch_template_ncc.format(datasets="_".join(group), metric=metric, gamma=gamma)
                with open(f"ncc_{first_dataset}_{last_dataset}_{metric}_{gamma}.sbatch", "w") as f:
                    f.write(sbatch_content_ncc)
        else:
            sbatch_content_ncc = sbatch_template_ncc.format(datasets="_".join(group), metric=metric, gamma="")
            with open(f"ncc_{first_dataset}_{last_dataset}_{metric}.sbatch", "w") as f:
                f.write(sbatch_content_ncc)

    for n_cluster in n_clusters:
        for metric in distance_metrics:
            if metric == 'softdtw':
                for gamma in gamma_values:
                    sbatch_content_clustering = sbatch_template_clustering.format(datasets="_".join(group), n_cluster=n_cluster, metric=metric, gamma=gamma)
                    with open(f"clustering_{first_dataset}_{last_dataset}_{n_cluster}_{metric}_{gamma}.sbatch", "w") as f:
                        f.write(sbatch_content_clustering)
            else:
                sbatch_content_clustering = sbatch_template_clustering.format(datasets="_".join(group), n_cluster=n_cluster, metric=metric, gamma="")
                with open(f"clustering_{first_dataset}_{last_dataset}_{n_cluster}_{metric}.sbatch", "w") as f:
                    f.write(sbatch_content_clustering)