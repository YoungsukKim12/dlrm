import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
output_dir = './tt_idx_histogram'
os.makedirs(output_dir, exist_ok=True)

# Load the file
file_path = 'tt_idx_profiling.sh'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize a dictionary to store tables
tables = defaultdict(list)

# Define regex patterns
table_pattern = re.compile(r'\[Table (\d+)\]')
array_pattern = re.compile(r'array\(\[([0-9, ]+)\]\)')

current_table = None

# Parse the file
for line in lines:
    table_match = table_pattern.match(line)
    if table_match:
        current_table = f'Table {table_match.group(1)}'
    else:
        arrays = array_pattern.findall(line)
        if arrays:
            arrays = [np.array(list(map(int, arr.split(', ')))) for arr in arrays]
            tables[current_table].append(arrays)

# Plot the data
for table, elements in tables.items():
    first_value_counts = [defaultdict(int) for _ in range(3)]

    for element in elements:
        for i in range(3):
            first_value_counts[i][element[i][0]] += 1

    for i, counts in enumerate(first_value_counts):
        plt.figure()
        plt.bar(counts.keys(), counts.values(), color='black')
        plt.xlabel('Index', color='black')
        plt.ylabel('Count', color='black')
        plt.title(f'Indices Distribution of Core {i+1} in {table}', color='black')
        plt.xticks(rotation=45, color='black')
        plt.yticks(color='black')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['top'].set_color('black') 
        plt.gca().spines['right'].set_color('black')
        plt.gca().spines['left'].set_color('black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{table}_core_{i+1}.png'), transparent=False)
        plt.close()

print("Done")
