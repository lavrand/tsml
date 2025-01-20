import os
import numpy as np

# datasets = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
            # 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction',
            # 'DistalPhalanxOutlineCorrect', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxTW', 'Earthquakes',
            # 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
            # 'Fish', 'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
            # 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
            # 'Mallat', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxOutlineAgeGroup',
            # 'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFatalECGThorax1', 'NonInvasiveFatalECGThorax2', 'OliveOil',
            # 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineCorrect',
            # 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim',
            # 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
            # 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2',
            # 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
            # 'UWaveGestureLibraryZ', 'UWaveGestureLibraryAll', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass',
            # 'Yoga', 'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'Cricket',
            # 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection', 'FingerMovements',
            # 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST',
            # 'MotorImagery', 'NATOPS', 'PenDigits', 'PEMS-SF', 'Phoneme', 'RacketSports', 'SelfRegulationSCP1',
            # 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

datasets = ['ArticularyWordRecognition', 'BasicMotions', 'CinCECGTorso', 'Cricket',
     'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'FaceDetection',
     'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'Libras',
     'LSST', 'MotorImagery', 'NATOPS', 'NonInvasiveFatalECGThorax1',
     'NonInvasiveFatalECGThorax2', 'PEMS-SF', 'PenDigits', 'RacketSports',
     'SelfRegulationSCP1', 'SelfRegulationSCP2', 'StandWalkJump', 'StarLightCurves',
     'UWaveGestureLibrary']

# datasets_grouped = [
#     ['ArticularyWordRecognition', 'BasicMotions', 'CinCECGTorso', 'Cricket',
#      'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'FaceDetection',
#      'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'Libras'],
#
#     ['LSST', 'MotorImagery', 'NATOPS', 'NonInvasiveFatalECGThorax1',
#      'NonInvasiveFatalECGThorax2', 'PEMS-SF', 'PenDigits', 'RacketSports',
#      'SelfRegulationSCP1', 'SelfRegulationSCP2', 'StandWalkJump', 'StarLightCurves',
#      'UWaveGestureLibrary']
# ]

# ['SmoothSubspace', 'Chinatown', 'ItalyPowerDemand', 'MelbournePedestrian', 'Crop', 'SyntheticControl', 'SonyAIBORobotSurface2',
#   'SonyAIBORobotSurface1', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'MiddlePhalanxOutlineAgeGroup',
#     'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'PhalangesOutlinesCorrect', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect',
#       'ProximalPhalanxTW', 'TwoLeadECG', 'MoteStrain', 'ECG200', 'ElectricDevices', 'MedicalImages', 'BME', 'CBF', 'SwedishLeaf', 'TwoPatterns',
#         'FaceAll', 'FacesUCR', 'ECGFiveDays', 'ECG5000', 'Plane', 'PowerCons', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale',
#           'GunPointOldVersusYoung', 'UMD', 'Wafer', 'ChlorineConcentration', 'Adiac', 'Fungi', 'Wine', 'Strawberry', 'ArrowHead', 'InsectWingbeatSound',
#             'FiftyWords', 'WordSynonyms', 'Trace', 'ToeSegmentation1', 'Coffee', 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'CricketX',
#               'CricketY', 'CricketZ', 'FreezerRegularTrain', 'FreezerSmallTrain', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
#                 'Lightning7', 'ToeSegmentation2', 'DiatomSizeReduction', 'FaceFour', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
#                   'PickupGestureWiimoteZ', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'ShakeGestureWiimoteZ', 'Symbols', 'Yoga',
#                     'OSULeaf', 'Ham', 'Meat', 'GesturePebbleZ1', 'GesturePebbleZ2', 'Fish', 'Beef', 'FordA', 'FordB', 'ShapeletSim', 'BeetleFly',
#                       'BirdChicken', 'Earthquakes', 'Herring', 'ShapesAll', 'OliveOil', 'Car', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain',
#                         'Lightning2', 'Computers', 'LargeKitchenAppliances', 'RefrigerationDevices', 'ScreenType', 'SmallKitchenAppliances',
#                           'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'Worms', 'WormsTwoClass', 'UWaveGestureLibraryAll', 'Mallat',
#                             'MixedShapesSmallTrain', 'Phoneme', 'StarlightCurves', 'Haptics', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'PLAID',
#                               'ACSF1', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'CinCECGtorso', 'EthanolLevel', 'InlineSkate',
#                                 'HouseTwenty', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'HandOutlines', 'Rock']

# datasets = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration']

# datasets = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly']

N_PARALLEL_DATASETS = 26

# Split the datasets into N_PARALLEL_DATASETS groups
dataset_groups = np.array_split(datasets, N_PARALLEL_DATASETS)

# datasets = ['CBF']
k_values = [1, 3, 5]
distance_metrics = ['euclidean', 'dtw', 'softdtw']
# gamma_values = [1]
gamma_values = [0.1, 1, 10]
n_clusters = [3]

sbatch_template_knn = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-23:50:00
#SBATCH --job-name knn_{k}_{metric}_{gamma}
#SBATCH --output /cs_storage/andreyl/pancake/knn_{k}_{metric}_{gamma}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=FAIL
#SBATCH --mem=4G
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

for group in dataset_groups:
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

    # for metric in distance_metrics:
    #     if metric == 'softdtw':
    #         for gamma in gamma_values:
    #             sbatch_content_ncc = sbatch_template_ncc.format(datasets="_".join(group), metric=metric, gamma=gamma)
    #             with open(f"ncc_{first_dataset}_{last_dataset}_{metric}_{gamma}.sbatch", "w") as f:
    #                 f.write(sbatch_content_ncc)
    #     else:
    #         sbatch_content_ncc = sbatch_template_ncc.format(datasets="_".join(group), metric=metric, gamma="")
    #         with open(f"ncc_{first_dataset}_{last_dataset}_{metric}.sbatch", "w") as f:
    #             f.write(sbatch_content_ncc)

    # for n_cluster in n_clusters:
    #     for metric in distance_metrics:
    #         if metric == 'softdtw':
    #             for gamma in gamma_values:
    #                 sbatch_content_clustering = sbatch_template_clustering.format(datasets="_".join(group), n_cluster=n_cluster, metric=metric, gamma=gamma)
    #                 with open(f"clustering_{first_dataset}_{last_dataset}_{n_cluster}_{metric}_{gamma}.sbatch", "w") as f:
    #                     f.write(sbatch_content_clustering)
    #         else:
    #             sbatch_content_clustering = sbatch_template_clustering.format(datasets="_".join(group), n_cluster=n_cluster, metric=metric, gamma="")
    #             with open(f"clustering_{first_dataset}_{last_dataset}_{n_cluster}_{metric}.sbatch", "w") as f:
    #                 f.write(sbatch_content_clustering)