import os
import itertools

# Define the different values for each hyperparameter
learning_rates = [0.01, 0.001]
episodes_list = [100, 200]
epsilon_decays = [0.7, 0.8]

# Create all combinations of hyperparameters
combinations = list(itertools.product(learning_rates, episodes_list, epsilon_decays))

for lr, ep, ed in combinations:
    command = f"python main_vf.py --train --learning_rate {lr} --episodes {ep} --epsilon_decay {ed}"
    os.system(command)