import os
import numpy as np

# N_PARALLEL_DATASETS = 113
N_PARALLEL_DATASETS = 18

datasets = [
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'Crop',
    'DodgerLoopDay',
    'DodgerLoopGame',
    'DodgerLoopWeekend',
    'ElectricDevices',
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