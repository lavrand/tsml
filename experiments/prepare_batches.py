import os

# datasets = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFatalECGThorax1', 'NonInvasiveFatalECGThorax2', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'UWaveGestureLibraryAll', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga', 'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS', 'PenDigits', 'PEMS-SF', 'Phoneme', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']
datasets = ['CBF']
k_values = [1, 3, 5]
distance_metrics = ['euclidean', 'dtw', 'softdtw']
gamma_values = [0.1, 1, 10]
n_clusters = [2, 3, 4]

sbatch_template_knn = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 1-23:50:00
#SBATCH --job-name knn_{dataset}_{k}_{metric}_{gamma}
#SBATCH --output /cs_storage/andreyl/pancake/knn_{dataset}_{k}_{metric}_{gamma}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --tasks=1

module load anaconda
source activate new_env

python knn.py --dataset {dataset} --k {k} --metric {metric} --gamma {gamma}
"""

sbatch_template_ncc = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 1-23:50:00
#SBATCH --job-name ncc_{dataset}_{metric}_{gamma}
#SBATCH --output /cs_storage/andreyl/pancake/ncc_{dataset}_{metric}_{gamma}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --tasks=1

module load anaconda
source activate new_env

python ncc.py --dataset {dataset} --metric {metric} --gamma {gamma}
"""

sbatch_template_clustering = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 1-23:50:00
#SBATCH --job-name clustering_{dataset}_{n_clusters}_{metric}_{gamma}
#SBATCH --output /cs_storage/andreyl/pancake/clustering_{dataset}_{n_clusters}_{metric}_{gamma}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --tasks=1

module load anaconda
source activate new_env

python clustering.py --dataset {dataset} --n_clusters {n_clusters} --metric {metric} --gamma {gamma}
"""

for dataset in datasets:
    for k in k_values:
        for metric in distance_metrics:
            for gamma in gamma_values:
                sbatch_content_knn = sbatch_template_knn.format(dataset=dataset, k=k, metric=metric, gamma=gamma)
                with open(f"knn_{dataset}_{k}_{metric}_{gamma}.sbatch", "w") as f:
                    f.write(sbatch_content_knn)

                sbatch_content_ncc = sbatch_template_ncc.format(dataset=dataset, metric=metric, gamma=gamma)
                with open(f"ncc_{dataset}_{metric}_{gamma}.sbatch", "w") as f:
                    f.write(sbatch_content_ncc)

    for n_clusters in n_clusters:
        for metric in distance_metrics:
            for gamma in gamma_values:
                sbatch_content_clustering = sbatch_template_clustering.format(dataset=dataset, n_clusters=n_clusters, metric=metric, gamma=gamma)
                with open(f"clustering_{dataset}_{n_clusters}_{metric}_{gamma}.sbatch", "w") as f:
                    f.write(sbatch_content_clustering)