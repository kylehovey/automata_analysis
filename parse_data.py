from tqdm import tqdm
import numpy as np
import csv
import os

if __name__ == "__main__":
    rules_dir = "./rules"
    target_file_out = './target.npz'
    data_file_out = './rule_data.npz'
    diff_file_out = './rule_diff.npz'
    target = []
    average_data = []
    diff_data = []
    rule_dirs = os.listdir(rules_dir)
    rule_numbers = sorted(map(int, rule_dirs))

    for rule_number in tqdm(rule_numbers):
        file_path = os.path.join(rules_dir, str(rule_number), 'trials.csv')

        with open(file_path, 'r') as trials_file:
            reader = csv.reader(trials_file)
            trial_data = [[int(size) for size in row[1:-1]] for row in reader]
            matrix = np.array(trial_data)
            trials_average = matrix.mean(axis=0)
            target.append(rule_number)
            average_data.append(trials_average)
            diff_data.append(np.diff(trials_average))

    with open(target_file_out, 'wb') as out_file:
        np.save(out_file, np.array(target))

    with open(data_file_out, 'wb') as out_file:
        np.save(out_file, np.array(average_data))

    with open(diff_file_out, 'wb') as out_file:
        np.save(out_file, np.array(diff_data))
