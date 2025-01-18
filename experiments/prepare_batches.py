import os
import numpy as np

# datasets = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
#             'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction',
#             'DistalPhalanxOutlineCorrect', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxTW', 'Earthquakes',
#             'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
#             'Fish', 'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
#             'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
#             'Mallat', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxOutlineAgeGroup',
#             'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFatalECGThorax1', 'NonInvasiveFatalECGThorax2', 'OliveOil',
#             'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineCorrect',
#             'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim',
#             'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
#             'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2',
#             'Trace', 'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
#             'UWaveGestureLibraryZ', 'UWaveGestureLibraryAll', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass',
#             'Yoga', 'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'Cricket',
#             'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection', 'FingerMovements',
#             'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST',
#             'MotorImagery', 'NATOPS', 'PenDigits', 'PEMS-SF', 'Phoneme', 'RacketSports', 'SelfRegulationSCP1',
#             'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']
datasets = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration']
N_PARALLEL_DATASETS = 8

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
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --tasks=1

module load anaconda
source activate new_env3

bash run_knn.sh {datasets} {k} {metric} {gamma}
"""

sbatch_template_ncc = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-23:50:00
#SBATCH --job-name ncc_{metric}_{gamma}
#SBATCH --output /cs_storage/andreyl/pancake/ncc_{metric}_{gamma}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --tasks=1

module load anaconda
source activate new_env3

bash run_ncc.sh {datasets} {metric} {gamma}
"""

sbatch_template_clustering = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-23:50:00
#SBATCH --job-name clustering_{n_cluster}_{metric}_{gamma}
#SBATCH --output /cs_storage/andreyl/pancake/clustering_{n_cluster}_{metric}_{gamma}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --tasks=1

module load anaconda
source activate new_env3

bash run_clustering.sh {datasets} {n_cluster} {metric} {gamma}
"""

for group in dataset_groups:
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

    for group in dataset_groups:
        group = list(group)  # Convert array back to list
        first_dataset = group[0]
        last_dataset = group[-1]
        for metric in distance_metrics:
            if metric == 'softdtw':
                for gamma in gamma_values:
                    sbatch_content_ncc = sbatch_template_ncc.format(datasets="_".join(group), metric=metric,
                                                                    gamma=gamma)
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
                        sbatch_content_clustering = sbatch_template_clustering.format(datasets="_".join(group),
                                                                                      n_cluster=n_cluster,
                                                                                      metric=metric, gamma=gamma)
                        with open(f"clustering_{first_dataset}_{last_dataset}_{n_cluster}_{metric}_{gamma}.sbatch",
                                  "w") as f:
                            f.write(sbatch_content_clustering)
                else:
                    sbatch_content_clustering = sbatch_template_clustering.format(datasets="_".join(group),
                                                                                  n_cluster=n_cluster, metric=metric,
                                                                                  gamma="")
                    with open(f"clustering_{first_dataset}_{last_dataset}_{n_cluster}_{metric}.sbatch", "w") as f:
                        f.write(sbatch_content_clustering)