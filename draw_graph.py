import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
import os
import math
import pickle

def hex_to_decimal(val):
    if isinstance(val, str):
        return int(val, 16) 
    return val

def Q_table_locality_profiler(table_profiles, hot_q_ratio):
    '''
        table_profiles : list of 2D numpy array
        exmaple) of table (2D numpy array)
            R1 R2 R3 R4
        Q1  2  3  1  2
        Q2  9  1  8  4
        Q3  7  2  5  6

    '''
    print(table_profiles)
    q_vec_hit_container = [[] for _ in range(len(table_profiles))]
    for i, table in enumerate(table_profiles):
        for entry in table:
            q_vec_hit = np.sum(entry)
            q_vec_hit_container[i].append(q_vec_hit)

    q_vec_hit_container = np.array(q_vec_hit_container)        
    q_vec_hit_container = q_vec_hit_container.reshape(-1)
    q_vec_hit_ratio_container = np.sort(q_vec_hit_container) / np.sum(q_vec_hit_container)

    accumulation_array = []
    accumulate_hit_ratio = 0
    iter_idx = 0

    while accumulate_hit_ratio < hot_q_hit_ratio:
        accumulation_array.append(q_vec_hit_ratio[iter_idx])
        accumulate_hit_ratio += q_vec_hit_ratio[iter_idx]
        iter_idx += 1

    return accumulation_array

def R_table_locality_profiler(table_profiles):
    r_vec_hit_container = [[] for _ in range(len(table_profiles))]
    for i, table in enumerate(table_profiles):
        r_table_profile = np.transpose(table)
        for entry in table:
            r_vec_hit = np.sum(r_table_profile[entry])
            r_vec_hit_container[i].append(r_vec_hit)

    return r_vec_hit_container        

def draw_original_locality_graph(vector_accumulation_array):
    fig = plt.figure(figsize=(6,4))
    plt.ylabel("Cumulative Access Rate (%)")
    plt.xlabel("Original Vectors Sorted by Access")

    plt.bar(vector_accumulation_array, color='#ed7d31', width = 0.4, edgecolor='k')
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 100))
    plt.tight_layout()
    plt.savefig("./graphs/vector_locality", dpi=300)

def draw_q_vector_locality_graph(q_vector_accumulation_array, collision):
    fig = plt.figure(figsize=(6,4))
    plt.ylabel("Cumulative Access Rate (%)")
    plt.xlabel("Q Vectors Sorted by Access")

    plt.bar(q_vector_accumulation_array, color='#ed7d31', width = 0.4, edgecolor='k')
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 100))
    plt.tight_layout()
    plt.savefig("./graphs/q_vector_locality", dpi=300)

def draw_r_vector_locality_graph(r_vector_access_array, collision):
    r_vectors = ["r vec "+str(i) for i in range(len(r_vector_access))]
    fig = plt.figure(figsize=(5,4))
    plt.ylabel("Access rate (%)")
    plt.xlabel("R vectors")

    plt.bar(r_vectors, r_vector_access_array, color='#4472c4', width = 0.2, align='center', edgecolor='k')
    plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 40, 10))
    plt.margins(0.15)
    plt.tight_layout()
    plt.savefig("./graphs/r_vector_locality", dpi=300)

def profile_hot_vector_variations(profile, collisions):
    total_hot_vectors = []
    for collision in collisions:
        combined_profile = np.vstack(profile)
        q_vec_access_profile = np.sum(combined_array, axis=1)
        sorted_indices = np.argsort(row_sums)
        sorted_sums = row_sums[sorted_indices]
        cumsum = np.cumsum(sorted_sums)
        limit = 0.8 * cumsum[-1]
        required_indices = np.where(cumsum <= limit)[0]
        total_hot_vectors.append(required_indices)

    return total_hot_vectors

def profile_azavu(collisions, nrows=20000):
    print('reading azavu data...')
    df = pd.read_csv("train", nrows=nrows)
    df = df.drop(columns=["id", "click"])
    df = df.applymap(hex_to_decimal)
    categorical_features = {category: df[category].tolist() for category in df.columns}
    datapath = "./avazu_profile.pickle"
    profile_size_datapath = "./profile_size.pickle"
    category_ranges = []
    if os.path.exists(profile_size_datapath):
        with open(profile_size_datapath, 'rb') as loadfile:
            category_ranges = pickle.load(loadfile)

    print("creating QR table...")
    # init qr profile
    QR_profile_table = [[] for _ in range(len(collisions))]
    for i, (name, category) in enumerate(categorical_features.items()):
        if len(category_ranges) == len(categorical_features):
            category_range = category_ranges[i]
        else:
            category_range = np.max(category)
            category_ranges.append(category_range)
        for j, collision in enumerate(collisions):
            q_range = math.ceil(category_range / collision) + 1
            qr_profiles = np.zeros((q_range, collision))
            QR_profile_table[j].append(qr_profiles)

    print("filling QR table...")
    for index, row in df.iterrows():
        for col_index, column in enumerate(df.columns):
            value = row[column]
            for j, collision in enumerate(collisions):
                QR_profile_table[j][col_index][value // collision][value % collision] += 1

    with open(profile_size_datapath, 'wb') as savefile:
        pickle.dump(category_ranges, savefile)

    return QR_profile_table

def save_criteo_kaggle_distribution(collisions):
    for collision in collisions:
        criteo_kaggle_profile_savefile = './savedata/profile_collision_%d.pickle' % collision
        profiles = None
        if not os.path.exists(savefile):
            print('collision %d not found.please run dlrm first!' % collision)
            sys.exit()
        else:
            with open(savefile, 'rb') as wf:
                prof_table = pickle.load(wf)

        q_table_prof = Q_table_locality_profiler(prof_table, 0.8)
        r_table_prof = R_table_locality_profiler(prof_table, 0.8)
        draw_q_vector_locality_graph(q_table_prof, collision)
        draw_r_vector_locality_graph(r_table_prof, collision)

def save_azavu_distribution(collisions):
    #     draw_original_locality_graph(prof_table)
    prof_tables = profile_azavu(collision)
    for i, collision in enumerate(collisions):
        prof_table = prof_tables[i]
        q_table_prof = Q_table_locality_profiler(prof_table, 0.8)
        r_table_prof = R_table_locality_profiler(prof_table, 0.8)
        draw_q_vector_locality_graph(q_table_prof, collision)
        draw_r_vector_locality_graph(r_table_prof, collision)

if __name__ == "__main__":
    if not os.path.exists("./graphs"):
        os.mkdir("./graphs")

    collisions = [4, 8, 16, 32]
    save_azavu_distribution(collisions)
    save_criteo_kaggle_distribution(collisions)