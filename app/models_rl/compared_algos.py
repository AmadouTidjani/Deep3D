import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd

# Define the hyperparameter combinations
learning_rates = [0.01, 0.001]
episodes_list = [100, 200]
epsilon_decays = [0.7, 0.8]

# Store the results
results = {}

# Load the .npy files
for lr in learning_rates:
    for ep in episodes_list:
        for ed in epsilon_decays:
            prefix = f"save/lr{lr}_ep{ep}_ed{ed}"
            scores = np.load(f"{prefix}_scores.npy")
            times = np.load(f"{prefix}_times.npy")
            avg_score = np.mean(scores)
            total_time = np.sum(times)
            results[prefix] = {'scores': scores, 'times': times, 'avg_score': avg_score, 'total_time': total_time}

# Plot comparison of scores
plt.figure(figsize=(10, 6))
for key, value in results.items():
    plt.plot(value['scores'], label=f"Model {key}")
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Comparison of Scores Across Models')
plt.legend()
plt.savefig(f'save/reward_algos.png')

# Plot comparison of times
plt.figure(figsize=(10, 6))
for key, value in results.items():
    plt.plot(value['times'], label=f"Model {key}")
plt.xlabel('Episodes')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Training Times Across Models')
plt.legend()
plt.savefig(f'save/time_algos.png')

# Create a DataFrame to summarize the average scores and total times
summary_table = pd.DataFrame(columns=['Model', 'Average Score', 'Total Time'])
for key, value in results.items():
    summary_table = summary_table.append({
        'Model': key,
        'Average Score': value['avg_score'],
        'Total Time': value['total_time']
    }, ignore_index=True)

# Print the summary table
print(summary_table)

# Save the summary table as a CSV file (optional)
summary_table.to_csv('save/model_comparison_summary.csv', index=False)