import subprocess
import concurrent.futures

datasets = ['GesturePebbleZ1']
k_values = [1, 3, 5]
distance_metrics = ['euclidean', 'dtw', 'softdtw']
gamma_values = [0.1, 1, 10]
n_clusters = [4]

def run_experiment(command):
    subprocess.run(command)

commands = []

for dataset in datasets:
    for k in k_values:
        for metric in distance_metrics:
            if metric == 'softdtw':
                for gamma in gamma_values:
                    commands.append(['python3', 'knn.py', '--dataset', dataset, '--k', str(k), '--metric', metric, '--gamma', str(gamma)])
            else:
                commands.append(['python3', 'knn.py', '--dataset', dataset, '--k', str(k), '--metric', metric])

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(run_experiment, commands)

    # for metric in distance_metrics:
    #     if metric == 'softdtw':
    #         for gamma in gamma_values:
    #             # Run NCC experiment
    #             subprocess.run(['python3', 'ncc.py', '--dataset', dataset, '--metric', metric, '--gamma', str(gamma)])
    #     else:
    #         # Run NCC experiment
    #         subprocess.run(['python3', 'ncc.py', '--dataset', dataset, '--metric', metric])
    #
    # for n_cluster in n_clusters:
    #     for metric in distance_metrics:
    #         if metric == 'softdtw':
    #             for gamma in gamma_values:
    #                 # Run Clustering experiment
    #                 subprocess.run(['python3', 'clustering.py', '--dataset', dataset, '--n_clusters', str(n_cluster), '--metric', metric, '--gamma', str(gamma)])
    #         else:
    #             # Run Clustering experiment
    #             subprocess.run(['python3', 'clustering.py', '--dataset', dataset, '--n_clusters', str(n_cluster), '--metric', metric])