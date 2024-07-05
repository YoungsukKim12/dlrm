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
        # print(q.shape)
        # print(MyProfiler.table_profiles[i][q,r])
        MyProfiler.table_profiles[i][q,r] += 1
        # print(MyProfiler.table_profiles[i][q,r])
        # print('next batch')

def total_compressed_embs():
    tot_embs = 0

    for table in MyProfiler.table_profiles:
        tot_embs += table.shape[0]
        tot_embs += table.shape[1]
    return tot_embs

def save_profile_result(path, dataset, collision):
    savefile = f'./{path}/{dataset}/profile_collision_{collision}.pickle'
    profiles = MyProfiler.table_profiles
    with open(savefile, 'wb') as sf:
        pickle.dump(profiles, sf)

def load_profile_result(path, dataset, collision):
    savefile = f'{path}/{dataset}/profile_collision_{collision}.pickle'
    profiles = None
    print(savefile)
    if not os.path.exists(savefile):
        print('please run dlrm first!')
        sys.exit()
    else:
        with open(savefile, 'rb') as wf:
            profiles = pickle.load(wf)            

    return profiles

def load_criteo_train_data(path='./savedata/', dataset='kaggle'):
    print('loading train data...')
    train_data = None
    train_data_savefile= os.path.join(path, dataset, "train_data.pickle")
    if not os.path.exists(train_data_savefile):
        print("read from kaggle")
        if dataset == 'kaggle':
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
        else:
            print("read from terabyte")
            train_data = dp.CriteoDataset(
                "terabyte",
                10000000,
                0.0,
                "total",
                "train",
                "./terabyte_input/day",
                "./terabyte_input/terabyte_processed.npz",
                True,
                False
            )
        with open(train_data_savefile, 'wb') as savefile:
                pickle.dump(train_data, savefile)
        pass
    else:
        with open(train_data_savefile, 'rb') as loadfile:
            train_data = pickle.load(loadfile)

    print('train data load complete!')
    return train_data


def get_physical_address(addr_translator, using_bg_map, table_index, vec_index, is_r_vec=False, collision_idx=0):
    in_HBM, emb_physical_addr = addr_translator.physical_address_translation(table_index, vec_index, is_r_vec, collision_idx)
    return in_HBM, emb_physical_addr

def write_trace_line(wf, device, physical_addr, vec_type, total_burst, table_idx=0, vec_idx=0):
    for k in range(total_burst):
        wf.write(f"{device} {physical_addr+64*k} {vec_type} {k} {table_idx} {vec_idx}\n")

def merge_kaggle_and_terabyte_data(
        train_data_path, 
        collision,
        terabyte_embedding_profiles,
        kaggle_duplicate_on_merge=1,
    ):

    kaggle_train_data = load_criteo_train_data(train_data_path, dataset='kaggle')
    kaggle_embedding_profiles = load_profile_result(path=train_data_path, dataset='kaggle', collision=collision)
    merged_profile = [*terabyte_embedding_profiles]
    for i in range(kaggle_duplicate_on_merge):
        total_access = 0
        print("kaggle profile len : ", len(kaggle_embedding_profiles))
        for j, prof_per_table in enumerate(kaggle_embedding_profiles):
            total_access += np.sum(prof_per_table)
        for j, prof_per_table in enumerate(kaggle_embedding_profiles):
            kaggle_embedding_profiles[j] = kaggle_embedding_profiles[j] / total_access / (kaggle_duplicate_on_merge+1)

        merged_profile = [*merged_profile, *kaggle_embedding_profiles]
    return kaggle_train_data, merged_profile

def write_trace_file(
        embedding_profile_savefile='./profile.pickle',
        train_data=None,
        collisions=4,
        vec_size=64,
        using_bg_map=False,
        called_inside_DLRM=False,
        dataset='kaggle',
        merge_kaggle_and_terabyte=True,
        kaggle_duplicate_on_merge=1
    ):
    train_data_path = './savedata'

    print('writing trace file...')
    if called_inside_DLRM:
        save_profile_result(train_data_path, dataset, collisions)
        embedding_profiles = MyProfiler.table_profiles
    else:
        embedding_profiles = load_profile_result(train_data_path, dataset, collisions)

    print('save & load complete')

    total_access = 0
    for i, prof_per_table in enumerate(embedding_profiles):
        total_access += np.sum(prof_per_table)
    for i, prof_per_table in enumerate(embedding_profiles):
        embedding_profiles[i] = embedding_profiles[i] / total_access / (kaggle_duplicate_on_merge+1)
    # print("terabyte profile len : ", len(embedding_profiles))

    if dataset == 'Terabyte' and merge_kaggle_and_terabyte:
        print('merging two datasets...')
        kaggle_train_data, merged_profile = merge_kaggle_and_terabyte_data(
                                                train_data_path=train_data_path,
                                                collision=collisions,
                                                terabyte_embedding_profiles=embedding_profiles,
                                                kaggle_duplicate_on_merge=kaggle_duplicate_on_merge,
                                            )
        embedding_profiles = merged_profile

    default_vec_size = 64
    total_burst = vec_size // default_vec_size
    total_data = len(train_data)

    addr_mappers = []

    # addr_mappers.append(BasicAddressTranslation(embedding_profiles, DIMM_size_gb=16, collisions=collisions, vec_size=vec_size, mapper_name="Basic"))
    # addr_mappers.append(RecNMPAddressTranslation(embedding_profiles, DIMM_size_gb=16, collisions=collisions, vec_size=vec_size, mapper_name="RecNMP")) # RecNMP
    # addr_mappers.append(TRiMAddressTranslation(embedding_profiles, DIMM_size_gb=16, collisions=collisions, bank_group_bits_naive=29, vec_size=vec_size, mapper_name="TRiM")) # TRiM
    # addr_mappers.append(HeteroBasicAddressTranslation(embedding_profiles, DIMM_size_gb=16, collisions=collisions, vec_size=vec_size, hot_vector_total_access=0.833, r_load_balance=False, mapper_name="SPACE")) # SPACE
    # addr_mappers.append(HeteroBasicAddressTranslation(embedding_profiles, DIMM_size_gb=16, collisions=collisions, vec_size=vec_size, hot_vector_total_access=0.8, r_load_balance=False, mapper_name="SPACE")) # SPACE
    # addr_mappers.append(HeteroBasicAddressTranslation(embedding_profiles, DIMM_size_gb=16, collisions=collisions, vec_size=vec_size, hot_vector_total_access=0.952, r_load_balance=False, mapper_name="HEAM")) #HEAM

    # for load balance test
    for i in [1]:
        for j in [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
            addr_mappers.append(RecNMPAddressTranslation(
                                            embedding_profiles=embedding_profiles, 
                                            DIMM_size_gb=i, 
                                            use_hot_access=False,
                                            use_hot_ratio=True,
                                            hot_vec_ratio=j,
                                            collisions=collisions, 
                                            vec_size=vec_size, 
                                            mapper_name=f"lb_test_{i}GB_{j}_hot"
                                        )
                                    )

    for addr_mapper in addr_mappers:
        mapper_name = addr_mapper.mapper_name()
        writefile = f'./{dataset}_{(mapper_name)}_{collisions}_col_{vec_size}_vec.txt'
        if merge_kaggle_and_terabyte:
            writefile = f'./{dataset}_{(mapper_name)}_{collisions}_col_{vec_size}_vec_{kaggle_duplicate_on_merge}_merge.txt'

        print("writing : ", writefile)
        with open(writefile, 'w') as wf:
            for i, data in enumerate(train_data):
                if i % 4000 == 0:
                    print(f"{i}/{total_data} trace processed")
                if i > 20000:
                    print('done writing tracing file')
                    break

                _, feat, _ = data
                if merge_kaggle_and_terabyte:
                    for i in range(kaggle_duplicate_on_merge):
                        _, kaggle_feat, _ = kaggle_train_data[random.randint(0, len(kaggle_train_data))]
                        feat = [*feat, *kaggle_feat]

                for table, emb in enumerate(feat):
                    q_emb = emb // collisions
                    r_emb = int(emb % collisions)

                    # write q vector
                    in_HBM, emb_physical_addr = get_physical_address(
                                                    addr_translator=addr_mapper, 
                                                    using_bg_map=using_bg_map, 
                                                    table_index=table, 
                                                    vec_index=q_emb,
                                                    is_r_vec=False,
                                                    collision_idx=r_emb
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
                    elif mapper_name == "HeteroBasic" or mapper_name == "HEAM":
                        device = "HBM" if in_HBM else "DIMM"
                        q_written_device = device
                        vec_type = "q" if collisions > 1 else "o"
                        write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)
                    elif mapper_name == "SPACE":
                        device = "HBM" if in_HBM else "DIMM"
                        q_written_device = device
                        vec_type = "q" if collisions > 1 else "o"
                        write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)
                    elif "lb_test" in mapper_name:
                        device = "HBM"
                        vec_type = "o"
                        if addr_mapper.is_hot_vector(table, q_emb):
                            continue
                        write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)

                    # write r vector                
                    if collisions > 1:
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
                        elif mapper_name == "HeteroBasic" or mapper_name == "HEAM":
                            # if q_written_device == "HBM":
                            device = "HBM"
                            vec_type = "r"
                            write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst, table, r_emb)
                        elif mapper_name == "SPACE":
                            # apply reduction locality to vectors in HBM. (do not add r trace for vector access in HBM) - This is for mildly applying reduction locality for convenience
                            if q_written_device == "DIMM":
                                device = "HBM"
                                vec_type = "r"
                                write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst, table, r_emb)
                        elif "lb_test" in mapper_name:
                            continue

                wf.write('\n')


if __name__ == "__main__":
    collisions = [8]
    vec_sizes = [64]
    dataset='kaggle'

    train_data = load_criteo_train_data('./savedata', dataset=dataset)

    for col in collisions:
        embedding_profile_savefile = f'./savedata/{dataset}/profile_collision_{col}.pickle'
        for vec_size in vec_sizes:
            print(f"Processing col={col}, vec_size={vec_size}")
            write_trace_file(embedding_profile_savefile=embedding_profile_savefile, 
                            train_data=train_data,
                            collisions=col,
                            vec_size=vec_size,
                            using_bg_map=False,
                            dataset=dataset,
                            merge_kaggle_and_terabyte=False,
                            kaggle_duplicate_on_merge=0
                            )

    # myutils.RunCacheSimulation(called_inside_DLRM=True, train_data=train_data, collision=args.qr_collisions)
