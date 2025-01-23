import os
import numpy as np

# N_PARALLEL_DATASETS = 113
N_PARALLEL_DATASETS = 4

datasets = [
    "ACSF1",
    "Adiac",
    "ArrowHead",
    "Beef",
    # "BeetleFly",
    # "BirdChicken",
    # "BME",
    # "Car",
    # "CBF",
    # "Chinatown",
    # "ChlorineConcentration",
    # "CinCECGTorso",
    # "Coffee",
    # "Computers",
    # "CricketX",
    # "CricketY",
    # "CricketZ",
    # "Crop",
    # "DiatomSizeReduction",
    # "DistalPhalanxOutlineCorrect",
    # "DistalPhalanxOutlineAgeGroup",
    # "DistalPhalanxTW",
    # "Earthquakes",
    # "ECG200",
    # "ECG5000",
    # "ECGFiveDays",
    # "ElectricDevices",
    # "EOGHorizontalSignal",
    # "EOGVerticalSignal",
    # "EthanolLevel",
    # "FaceAll",
    # "FaceFour",
    # "FacesUCR",
    # "FiftyWords",
    # "Fish",
    # "FordA",
    # "FordB",
    # "FreezerRegularTrain",
    # "FreezerSmallTrain",
    # "Fungi",
    # "GunPoint",
    # "GunPointAgeSpan",
    # "GunPointMaleVersusFemale",
    # "GunPointOldVersusYoung",
    # "Ham",
    # "HandOutlines",
    # "Haptics",
    # "Herring",
    # "HouseTwenty",
    # "InlineSkate",
    # "InsectEPGRegularTrain",
    # "InsectEPGSmallTrain",
    # "InsectWingbeatSound",
    # "ItalyPowerDemand",
    # "LargeKitchenAppliances",
    # "Lightning2",
    # "Lightning7",
    # "Mallat",
    # "Meat",
    # "MedicalImages",
    # "MiddlePhalanxOutlineCorrect",
    # "MiddlePhalanxOutlineAgeGroup",
    # "MiddlePhalanxTW",
    # "MixedShapesRegularTrain",
    # "MixedShapesSmallTrain",
    # "MoteStrain",
    # "NonInvasiveFetalECGThorax1",
    # "NonInvasiveFetalECGThorax2",
    # "OliveOil",
    # "OSULeaf",
    # "PhalangesOutlinesCorrect",
    # "Phoneme",
    # "PigAirwayPressure",
    # "PigArtPressure",
    # "PigCVP",
    # "Plane",
    # "PowerCons",
    # "ProximalPhalanxOutlineCorrect",
    # "ProximalPhalanxOutlineAgeGroup",
    # "ProximalPhalanxTW",
    # "RefrigerationDevices",
    # "Rock",
    # "ScreenType",
    # "SemgHandGenderCh2",
    # "SemgHandMovementCh2",
    # "SemgHandSubjectCh2",
    # "ShapeletSim",
    # "ShapesAll",
    # "SmallKitchenAppliances",
    # "SmoothSubspace",
    # "SonyAIBORobotSurface1",
    # "SonyAIBORobotSurface2",
    # "StarlightCurves",
    # "Strawberry",
    # "SwedishLeaf",
    # "Symbols",
    # "SyntheticControl",
    # "ToeSegmentation1",
    # "ToeSegmentation2",
    # "Trace",
    # "TwoLeadECG",
    # "TwoPatterns",
    # "UMD",
    # "UWaveGestureLibraryAll",
    # "UWaveGestureLibraryX",
    # "UWaveGestureLibraryY",
    # "UWaveGestureLibraryZ",
    # "Wafer",
    # "Wine",
    # "WordSynonyms",
    # "Worms",
    # "WormsTwoClass",
    # "Yoga",
]

# Split the datasets into N_PARALLEL_DATASETS groups
dataset_groups = np.array_split(datasets, N_PARALLEL_DATASETS)

# k_values = [1, 3, 5]
# distance_metrics = ['euclidean', 'dtw', 'softdtw']
# gamma_values = [0.1, 1, 10]
# n_clusters = [3]

sbatch_template_shapedtw_knn = """#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-23:50:00
#SBATCH --job-name shapedtw_knn_{shape_function}
#SBATCH --output /cs_storage/andreyl/pancake/shapedtw_knn_{shape_function}-id-%J.out
#SBATCH --mail-user=andreyl@post.bgu.ac.il
#SBATCH --mail-type=FAIL
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --tasks=1

module load anaconda
source activate tsml

bash shapedtw_knn.sh {datasets} {shape_function}
"""

shape_functions = ["raw", "hog1d"]

for group in dataset_groups:
    if len(group) == 0:
        continue
    group = list(group)  # Convert array back to list
    first_dataset = group[0]
    last_dataset = group[-1]
    for shape_function in shape_functions:
        sbatch_content_shapedtw_knn = sbatch_template_shapedtw_knn.format(datasets="_".join(group), shape_function=shape_function)
        with open(f"shapedtw_knn_{first_dataset}_{last_dataset}_{shape_function}.sbatch", "w") as f:
            f.write(sbatch_content_shapedtw_knn)

# sbatch_template_knn = """#!/bin/bash
# #SBATCH --partition main
# #SBATCH --time 6-23:50:00
# #SBATCH --job-name knn_{k}_{metric}_{gamma}
# #SBATCH --output /cs_storage/andreyl/pancake/knn_{k}_{metric}_{gamma}-id-%J.out
# #SBATCH --mail-user=andreyl@post.bgu.ac.il
# #SBATCH --mail-type=FAIL
# #SBATCH --mem=10G
# #SBATCH --cpus-per-task=4
# #SBATCH --tasks=1
#
# module load anaconda
# source activate tsml
#
# bash run_knn.sh {datasets} {k} {metric} {gamma}
# """
#
# sbatch_template_ncc = """#!/bin/bash
# #SBATCH --partition main
# #SBATCH --time 6-23:50:00
# #SBATCH --job-name ncc_{metric}_{gamma}
# #SBATCH --output /cs_storage/andreyl/pancake/ncc_{metric}_{gamma}-id-%J.out
# #SBATCH --mail-user=andreyl@post.bgu.ac.il
# #SBATCH --mail-type=FAIL
# #SBATCH --mem=4G
# #SBATCH --cpus-per-task=4
# #SBATCH --tasks=1
#
# module load anaconda
# source activate tsml
#
# bash run_ncc.sh {datasets} {metric} {gamma}
# """
#
# sbatch_template_clustering = """#!/bin/bash
# #SBATCH --partition main
# #SBATCH --time 6-23:50:00
# #SBATCH --job-name clustering_{n_cluster}_{metric}_{gamma}
# #SBATCH --output /cs_storage/andreyl/pancake/clustering_{n_cluster}_{metric}_{gamma}-id-%J.out
# #SBATCH --mail-user=andreyl@post.bgu.ac.il
# #SBATCH --mail-type=FAIL
# #SBATCH --mem=20G
# #SBATCH --cpus-per-task=10
# #SBATCH --tasks=1
#
# module load anaconda
# source activate tsml
#
# bash run_clustering.sh {datasets} {n_cluster} {metric} {gamma}
# """
#
# for group in dataset_groups:
#     if len(group) == 0:
#         continue
#     group = list(group)  # Convert array back to list
#     first_dataset = group[0]
#     last_dataset = group[-1]
#     for k in k_values:
#         for metric in distance_metrics:
#             if metric == 'softdtw':
#                 for gamma in gamma_values:
#                     sbatch_content_knn = sbatch_template_knn.format(datasets="_".join(group), k=k, metric=metric, gamma=gamma)
#                     with open(f"knn_{first_dataset}_{last_dataset}_{k}_{metric}_{gamma}.sbatch", "w") as f:
#                         f.write(sbatch_content_knn)
#             else:
#                 sbatch_content_knn = sbatch_template_knn.format(datasets="_".join(group), k=k, metric=metric, gamma="")
#                 with open(f"knn_{first_dataset}_{last_dataset}_{k}_{metric}.sbatch", "w") as f:
#                     f.write(sbatch_content_knn)

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