import os
import itertools
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description = 'Train or test neural net')
parser.add_argument('--test', dest = 'test', action = 'store_true', default = False)
parser.add_argument('--episodes', type=int, default=200, help=' Nb itérations ')

args = parser.parse_args()

results = {}

# Define the different values for each hyperparameter
learning_rates = [0.01, 0.001]
batch_size = [2, 4]
epsilon_decays = [0.5, 0.8]

# Create all combinations of hyperparameters
combinations = list(itertools.product(learning_rates, batch_size, epsilon_decays))

for lr, b, ed in combinations:
    if args.test:
        command = f"python main_vf.py --test --learning_rate {lr} --batch_size {b} --epsilon_decay {ed} --episodes {args.episodes}"
    else:
        command = f"python main_vf.py --train --learning_rate {lr} --batch_size {b} --epsilon_decay {ed} --episodes {args.episodes}"
    os.system(command)


# Load the .npy files
for lr in learning_rates:
    for b in batch_size:
        for ed in epsilon_decays:
            if args.test:
                prefix = f"save/test_lr{lr}_b{b}_ed{ed}"
            else:
                prefix = f"save/lr{lr}_b{b}_ed{ed}"

            scores = np.load(f"{prefix}_scores.npy")
            times = np.load(f"{prefix}_times.npy")
            avg_score = np.max(scores)
            total_time = np.sum(times)
            results[prefix] = {'scores': scores, 'times': times, 'avg_score': avg_score, 'total_time': total_time}

if args.test:
    pathTimes = f'save/test_time_algos.png'
    pathRewards = f'save/test_reward_algos.png'
    pathCSV = 'save/test_model_comparison_summary.csv'
else :
    pathTimes = f'save/train_time_algos.png'
    pathRewards = f'save/train_reward_algos.png'
    pathCSV = 'save/train_model_comparison_summary.csv'
    
# Plot comparison of scores
plt.figure(figsize=(10, 6))
for key, value in results.items():
    key = os.path.basename(key)
    plt.plot(value['scores'], label=f"Model {key}")
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Comparison of Scores Across Models')
plt.legend()
plt.savefig(pathRewards)

# Plot comparison of times
plt.figure(figsize=(10, 6))
for key, value in results.items():
    key = os.path.basename(key)
    plt.plot(value['times'], label=f"Model {key}")
plt.xlabel('Episodes')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Training Times Across Models')
plt.legend()
plt.savefig(pathTimes)

# Create a DataFrame to summarize the average scores and total times
summary_table = [] #pd.DataFrame(columns=['Model', 'Average Score', 'Total Time'])
for key, value in results.items():
    summary_table.append({
        'Model': key,
        'Average Score': value['avg_score'],
        'Total Time': value['total_time']})#, ignore_index=True)

# Print the summary table
summary_table = pd.DataFrame(summary_table)
print(summary_table)

# Save the summary table as a CSV file (optional)
summary_table.to_csv(pathCSV, index=False)