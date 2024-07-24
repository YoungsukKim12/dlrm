import re
from collections import defaultdict
import numpy as np

file_path = 'tt_table_profiling.sh'

table_data = defaultdict(list)

with open(file_path, 'r') as file:
    current_table = None
    current_data = []
    for line in file:
        line = line.strip()
        if line.startswith('[Table'):
            if current_table is not None:
                table_data[current_table].extend(current_data)
                current_data = []
            current_table = re.findall(r'\d+', line)[0]
        elif current_table:
            if line.startswith('[') and line.endswith(']'):
                line = line.replace('array', 'np.array')
                current_data.append(eval(line))

    if current_table and current_data:
        table_data[current_table].extend(current_data)

for table, data in table_data.items():
    print(f"[Table {table}]")
    for item in data:
        for array_item in item:
            print(array_item)
    print()
