import numpy as np
import math
import random
import pickle
import os
import dlrm_data_pytorch as dp
import sys
from ProactivePIMTranslation import ProactivePIMTranslation

def save_profile_result(path, dataset):
    savefile = f'./{path}/{dataset}/profile.pickle'
    profiles = MyProfiler.table_profiles
    with open(savefile, 'wb') as sf:
        pickle.dump(profiles, sf)

def load_profile_result(path, dataset):
    savefile = f'{path}/{dataset}/profile.pickle'
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
        save_profile_result(train_data_path, dataset)
        embedding_profiles = MyProfiler.table_profiles
    else:
        embedding_profiles = load_profile_result(train_data_path, dataset, collisions)

    print('save & load complete')

    total_access = 0
    for i, prof_per_table in enumerate(embedding_profiles):
        total_access += np.sum(prof_per_table)
    for i, prof_per_table in enumerate(embedding_profiles):
        embedding_profiles[i] = embedding_profiles[i] / total_access / (kaggle_duplicate_on_merge+1)

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
    addr_mappers.append(ProactivePIMTranslation(embedding_profiles, vec_size=vec_size, HBM_size_gb=4, is_QR=True, collisions=collisions, mapper_name="ProactivePIM_QR"))
    addr_mappers.append(ProactivePIMTranslation(embedding_profiles, vec_size=vec_size, HBM_size_gb=4, is_TT_Rec=True, rank=16, mapper_name="ProactivePIM_TT_Rec"))

    for addr_mapper in addr_mappers:
        mapper_name = addr_mapper.mapper_name()
        writefile = f'./{dataset}_{(mapper_name)}_{vec_size}_vec.txt'
        if merge_kaggle_and_terabyte:
            writefile = f'./{dataset}_{(mapper_name)}_{vec_size}_vec_{kaggle_duplicate_on_merge}_merge.txt'

        print("writing : ", writefile)
        with open(writefile, 'w') as wf:
            for i, data in enumerate(train_data):
                if i % 1000 == 0:
                    print(f"{i}/{total_data} trace processed")
                if i > 5000:
                    print('done writing tracing file')
                    break

                _, feat, _ = data
                if merge_kaggle_and_terabyte:
                    for i in range(kaggle_duplicate_on_merge):
                        _, kaggle_feat, _ = kaggle_train_data[random.randint(0, len(kaggle_train_data))]
                        feat = [*feat, *kaggle_feat]

                for table, emb in enumerate(feat):
                    physical_addr = addr_mapper.physical_translation(table, emb)
                    if "QR" in mapper_name:
                        device = "HBM"
                        vec_type = "o"
                        write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)
                    elif "TT_Rec" in mapper_name:
                        device = "HBM"
                        vec_type = "o"
                        write_trace_line(wf, device, emb_physical_addr, vec_type, total_burst)

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
