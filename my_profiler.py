import numpy as np
import math
import random
import pickle
import os
import dlrm_data_pytorch as dp
import sys
from address_translation import HeteroBasicAddressTranslation, BasicAddressTranslation, TRiMAddressTranslation, RecNMPAddressTranslation

class MyProfiler:

    table_profiles = []
    table_index = []

    @staticmethod
    def set_qr_profile(table, q_len, r_len):
        if table > 0:
            for i in range(table-MyProfiler.table_index[-1]-1):
                MyProfiler.table_profiles.append(np.zeros((1,1)))
        MyProfiler.table_index.append(table)
        qr_profiles = np.zeros((q_len,r_len))
        MyProfiler.table_profiles.append(qr_profiles)

    @staticmethod
    def record_qr_profile(i, q, r):
        q = q.detach().cpu().numpy()
        r = r.detach().cpu().numpy()
        MyProfiler.table_profiles[i][q,r] += 1

def total_compressed_embs():
    tot_embs = 0

    for table in MyProfiler.table_profiles:
        tot_embs += table.shape[0]
        tot_embs += table.shape[1]
    return tot_embs

def save_profile_result(collision):
    savefile = './savedata/profile_collision_%d.pickle' % collision
    profiles = MyProfiler.table_profiles
    with open(savefile, 'wb') as sf:
        pickle.dump(profiles, sf)

def load_profile_result(savefile):
    profiles = None
    if not os.path.exists(savefile):
        print('please run dlrm first!')
        sys.exit()
    else:
        with open(savefile, 'rb') as wf:
            profiles = pickle.load(wf)            

    return profiles

def load_train_data(train_data_savefile):
    print('loading train data...')
    if not os.path.exists(train_data_savefile):
        train_data = dp.CriteoDataset(
            "kaggle",
            -1,
            0.0,
            "total",
            "train",
            "./input/train.txt",
            "./input/kaggleAdDisplayChallenge_processed.npz",
            False,
            False
        )
        with open(train_data_savefile, 'wb') as savefile:
            pickle.dump(train_data, savefile)
    else:
        with open(train_data_savefile, 'rb') as loadfile:
            train_data = pickle.load(loadfile)

    print('train data load complete!')
    return train_data


def get_physical_address(addr_translator, using_bg_map, table_index, vec_index, is_r_vec=False):
    in_HBM, emb_physical_addr = addr_translator.physical_address_translation(table_index, vec_index, is_r_vec)
    return in_HBM, emb_physical_addr

def write_trace_line(wf, device, physical_addr, vec_type, total_burst):
    for k in range(total_burst):
        wf.write(f"{device} {physical_addr+64*k} {vec_type} {k}\n")

def write_trace_file(
        embedding_profile_savefile='./profile.pickle', 
        train_data=None, 
        collisions=4, 
        vec_size=64,
        using_bg_map=False,
        using_TRiM=False,
        called_inside_DLRM=False,
        dataset='Kaggle'
    ):
    if called_inside_DLRM:
        embedding_profiles = MyProfiler.table_profiles
        save_profile_result(collisions)
    else:
        embedding_profiles = load_profile_result(embedding_profile_savefile)

    default_vec_size = 64
    total_burst = vec_size // default_vec_size
    total_data = len(train_data)

    addr_mappers = []
    # addr_mappers.append(BasicAddressTranslation(embedding_profiles, DIMM_size_gb=4, collisions=collisions, vec_size=vec_size))
    # addr_mappers.append(RecNMPAddressTranslation(embedding_profiles, collisions=collisions, vec_size=vec_size)) # RecNMP
    addr_mappers.append(TRiMAddressTranslation(embedding_profiles, collisions=collisions, bank_group_bits_naive=27)) # TRiM
    # addr_mappers.append(HeteroBasicAddressTranslation(embedding_profiles, collisions=collisions, vec_size=vec_size, hot_vector_total_access=0.8)) # SPACE
    # addr_mappers.append(HeteroBasicAddressTranslation(embedding_profiles, collisions=collisions, vec_size=vec_size, hot_vector_total_access=0.9, mapper_name="HEAM")) #HEAM

    for addr_mapper in addr_mappers:
        mapper_name = addr_mapper.mapper_name()
        writefile = f'./traces/{dataset}_{(mapper_name)}_trace_col_{collisions}_vecsize_{vec_size}.txt'
        print("writing : ", writefile)
        with open(writefile, 'w') as wf:
            for i, data in enumerate(train_data):
                if i % 4096 == 0:
                    print(f"{i}/{total_data} trace processed")
                if i > 20000:
                    print('done writing tracing file')
                    break

                _, feat, _ = data
                for table, emb in enumerate(feat):
                    q_emb = emb // collisions

                    # write q vector
                    in_HBM, emb_physical_addr = get_physical_address(
                                                    addr_translator=addr_mapper, 
                                                    using_bg_map=using_bg_map, 
                                                    table_index=table, 
                                                    vec_index=q_emb,
                                                    is_r_vec=False
                                                )
                    if mapper_name == "Basic":
                        device = "DIMM"
                        vec_type = "o"
                        write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)
                    elif mapper_name == "RecNMP":
                        device = "DIMM"
                        vec_type = "o"
                        if addr_mapper.is_hot_vector(table, q_emb):
                            vec_type = 'h'
                        write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)
                    elif mapper_name == "TRiM":
                        device = "DIMM"
                        vec_type = "o"
                        if addr_mapper.is_hot_vector(table, q_emb):
                            vec_type = 'h'
                        write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)
                    elif mapper_name == "HeteroBasic":
                        device = "HBM" if in_HBM else "DIMM"
                        q_written_device = device
                        vec_type = "q" if collisions > 1 else "o"
                        write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)

                    # write r vector                
                    if collisions > 1:
                        r_emb = emb % collisions
                        in_HBM, emb_physical_addr = get_physical_address(
                                                        addr_translator=addr_mapper, 
                                                        using_bg_map=using_bg_map, 
                                                        table_index=table, 
                                                        vec_index=r_emb,
                                                        is_r_vec=True
                                                    )
                        if mapper_name == "Basic":
                            device = "DIMM"
                            vec_type = "o"
                            write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)
                        elif mapper_name == "RecNMP":
                            device = "DIMM"
                            vec_type = "h"
                            write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)
                        elif mapper_name == "TRiM":
                            device = "DIMM"
                            vec_type = "h"
                            write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)
                        elif mapper_name == "HeteroBasic":
                            if q_written_device == "HBM":
                                device = "HBM"
                                vec_type = "r"
                                write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)

                wf.write('\n')


if __name__ == "__main__":
    profiles = None
    collisions = [1]
    vec_sizes = [64, 128, 256, 512]

    train_data_savefile = './savedata/train_data.pickle'
    train_data = load_train_data(train_data_savefile)
    for col in collisions:
        embedding_profile_savefile = './savedata/profile_collision_%d.pickle' % col
        for vec_size in vec_sizes:
            print(f"Processing col={col}, vec_size={vec_size}")
            write_trace_file(embedding_profile_savefile=embedding_profile_savefile, 
                            train_data=train_data,
                            collisions=col,
                            vec_size=vec_size,
                            using_bg_map=False,
                            dataset='kaggle'
                            )

    # convertDicts = [{} for _ in range(26)]  # Initialize empty dictionaries for each categorical feature
    # npzfile = "./terabyte_input/day"
    # counts = np.zeros(26, dtype=np.int32)
    # for i in range(24):
    #     print(f'Loading day {i} data from {npzfile}_{i}.npz...')
    #     with np.load(f'./terabyte_input/day_{i}.npz') as data:
    #         # Assuming X_cat_t is the transposed categorical data
    #         X_cat_t = data["X_cat_t"]
    #         for j in range(26):  # Iterate over each categorical feature
    #             unique_values = np.unique(X_cat_t[j, :])  # Find unique values for each feature
    #             for value in unique_values:
    #                 convertDicts[j][value] = 1  # Update convertDicts

    #     # Convert indices in convertDicts to sequential numbers
    #     for j in range(26):
    #         unique_values = list(convertDicts[j].keys())
    #         convertDicts[j] = {old: new for new, old in enumerate(unique_values)}
    #         counts[j] = len(convertDict[j])

    # dict_file_j = d_path + d_file + "_fea_dict_{0}_tmp.npz".format(j)
    # if not path.exists(dict_file_j):
    #     np.savez_compressed(
    #         dict_file_j,
    #         unique=np.array(list(convertDicts[j]), dtype=np.int32)
    #     )
    # print("Saved convertDicts to file.")


    # convertDicts = [{} for _ in range(26)]
    # counts = np.zeros(26, dtype=np.int32)

    # for i in range(24):
    #     print("processing ", i)
    #     with np.load("./terabyte_input/day_{0}_processed.npz".format(i)) as f:
    #         X_cat = f["X_cat"]
    #     print(X_cat.shape[0])
    #     for k in range(X_cat.shape[0]):
    #         for j in range(26):
    #             print(X_cat[k][j])
    #             convertDicts[j][X_cat[k][j]] = 1
    #     for j in range(26):
    #         for m, x in enumerate(convertDicts[j]):
    #             convertDicts[j][x] = m
    #         counts[j] = len(convertDicts[j])
    #     print(counts)

    # count_file = "./terabyte_input/day_fea_count.npz"
    # np.savez_compressed(count_file, counts=counts)
