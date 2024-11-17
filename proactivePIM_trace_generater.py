import numpy as np
import math
import random
import pickle
import os
import dlrm_data_pytorch as dp
import sys
import cache_simulator as CacheSimulator
sys.path.insert(0, '..')
from proactivePIM_translation import ProactivePIMTranslation
from multi_hot import Multihot

class EmbTableProfiler:

    table_profiles = []
    table_index = []

    @staticmethod
    def set_table_profile(table_id, table_len):
        table_pf = []
        for i in range(table_len):
            table_pf.append(np.zeros(1))
        EmbTableProfiler.table_profiles.append(table_pf)

    @staticmethod
    def record_profile(table_id, vec_ids):
        for vec_id in vec_ids:
            EmbTableProfiler.table_profiles[table_id][vec_id] += 1


def save_profile_result(path, dataset):
    savefile = f'./{path}/{dataset}/profile.pickle'
    profiles = EmbTableProfiler.table_profiles
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

def write_trace_line(wf, device, physical_addr, command, total_burst, cache=None):
    if not cache == None:
        if not cache.access(physical_addr):
            wf.write(f"{device} {command} {physical_addr} {int(total_burst)} \n")
    else:
        wf.write(f"{device} {command} {physical_addr} {int(total_burst)} \n")

def data_move(write_file, device, addr_mapper, addr, cmp_addr, pim_level, burst):
    write_trace_line(write_file, device, addr, "RD", burst)
    transfer_addr, same_node = addr_mapper.map_to_same_node(pim_level, cmp_addr, addr, randomize_row=True)
    if not same_node:
        # write_trace_line(write_file, device, transfer_addr, "WR", burst)
        write_trace_line(write_file, device, transfer_addr, "DR", burst)

def merge_kaggle_and_terabyte_data(
        train_data_path, 
        collision,
        terabyte_embedding_profiles,
        kaggle_duplicate_on_merge=1,
    ):

    kaggle_train_data = load_criteo_train_data(train_data_path, dataset='kaggle')
    kaggle_embedding_profiles = load_profile_result(path=train_data_path, dataset='kaggle')
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
        embedding_profiles=None,
        train_data=None,
        called_inside_DLRM=False,
        dataset='kaggle',
        merge_kaggle_and_terabyte=True,
        kaggle_duplicate_on_merge=1,
        total_trace=10,
        collisions=4,
        tt_rank=16, 
        vec_size=64,
        batch_size=4,
        using_vp=False,
        table_prefetch=True,
        all_prefetch=False,
        using_subtable_mapping=True,
        addr_map={},
        cpu_baseline=False,
        cache=None,
        pim_level="bankgroup"
    ):

    if len(addr_map) != 6:
        print("please provide correct address mapping!")
        sys.exit()

    train_data_path = './savedata'
    total_batch = total_trace // batch_size

    print('writing trace file...')
    if called_inside_DLRM:
        save_profile_result(train_data_path, dataset)
        embedding_profiles = EmbTableProfiler.table_profiles
    else:
        if embedding_profiles == None:
            embedding_profiles = load_profile_result(train_data_path, dataset)

    print('save & load complete')

    total_access = 0
    for i, prof_per_table in enumerate(embedding_profiles):
        total_access += np.sum(prof_per_table)
    for i, prof_per_table in enumerate(embedding_profiles):
        embedding_profiles[i] = embedding_profiles[i] / total_access

    default_vec_size = 64
    total_burst = vec_size // default_vec_size
    tt_rec_burst = tt_rank * 4 // default_vec_size
    total_data = len(train_data)
    using_prefetch = all_prefetch or table_prefetch

    multi_hot = Multihot(
                    multi_hot_sizes=[80 for i in range(len(embedding_profiles))],
                    num_embeddings_per_feature=[len(table) for table in embedding_profiles],
                    batch_size=1,
                    collect_freqs_stats=False,
                    dist_type='pareto',
                    dataset=dataset
                )

    addr_mappers = []

    # # original trace
    # addr_mappers.append(
    #         ProactivePIMTranslation(
    #             embedding_profiles, 
    #             vec_size=vec_size, 
    #             HBM_size_gb=16.1, 
    #             is_QR=False,
    #             is_TT_Rec=False, 
    #             collisions=collisions, 
    #             using_prefetch=using_prefetch,
    #             using_subtable_mapping=using_subtable_mapping,
    #             addr_map=addr_map,
    #             pim_level=pim_level,
    #             mapper_name="ProactivePIM_Original"
    #         )
    # )

    # QR trace
    addr_mappers.append(
            ProactivePIMTranslation(
                embedding_profiles, 
                vec_size=vec_size, 
                HBM_size_gb=4, 
                is_QR=True, 
                collisions=collisions, 
                using_prefetch=using_prefetch,
                using_subtable_mapping=using_subtable_mapping,
                addr_map=addr_map,
                pim_level=pim_level,
                mapper_name="ProactivePIM_QR"
            )
    )

    # # TT-Rec trace
    addr_mappers.append(
            ProactivePIMTranslation(
                embedding_profiles, 
                vec_size=vec_size, 
                HBM_size_gb=4, 
                is_TT_Rec=True, 
                tt_rank=tt_rank, 
                using_prefetch=using_prefetch,
                using_subtable_mapping=using_subtable_mapping,
                using_gemv_dist=True,
                addr_map=addr_map,
                pim_level=pim_level,
                mapper_name="ProactivePIM_TT_Rec"
            )
    )

    print([len(table) for table in embedding_profiles])

    for addr_mapper in addr_mappers:
        mapper_name = addr_mapper.mapper_name()
        is_QR = True if "QR" in mapper_name else False
        is_TT_Rec = True if "TT_Rec" in mapper_name else False
        writefile = f'./{dataset}_{(mapper_name)}_vec_{vec_size}_prefetch_{using_prefetch}_mapping_{using_subtable_mapping}.txt'
        print("writing : ", writefile)

        with open(writefile, 'w') as wf:
            for i in range(len(train_data)//batch_size):
                batch_data = [feat for _, feat, _ in train_data[i*batch_size:(i+1)*batch_size]]
                batch_data = np.array(batch_data, dtype=np.int64)
                batch_data = np.transpose(batch_data)
                multi_hot_indices = multi_hot.make_new_batch(lS_i=batch_data, batch_size=batch_size)
                multi_hot_indices = np.transpose(multi_hot_indices)

                if i % 100 == 0:
                    print(f"{i}/{total_batch} trace processed")
                if i > total_batch:
                    break

                # prefetch only once
                if all_prefetch and i == 0:
                        for table in range(len(embedding_profiles)):
                            prefetch_addrs = addr_mapper.get_prefetch_physical_address(table)
                            if is_QR:
                                write_trace_line(wf, device, prefetch_addrs[i], "PR", total_burst)
                            elif is_TT_Rec:
                                write_trace_line(wf, device, prefetch_addrs[i], "PR", tt_rec_burst)

                for table, batch_embs in enumerate(multi_hot_indices):                 
                    # inital execution has no result to transfer // batch_size * 4 => four bank groups
                    transfers = 0 if i == 0 and table == 0 else batch_size * 4 
                    if using_prefetch:
                        prefetch_addrs = addr_mapper.get_prefetch_physical_address(table)
                        overhead = transfers if transfers > len(prefetch_addrs) else len(prefetch_addrs)
                        for i in range(overhead):
                            if i < transfers:
                                write_trace_line(wf, device, 0, "TR", total_burst)
                            if i < len(prefetch_addrs):
                                if is_QR:
                                    write_trace_line(wf, device, prefetch_addrs[i], "PR", total_burst)
                                elif is_TT_Rec:
                                    write_trace_line(wf, device, prefetch_addrs[i], "PR", tt_rec_burst)
                    else:
                        for i in range(transfers):
                            write_trace_line(wf, device, 0, "TR", total_burst)

                    for l, multi_embs in enumerate(batch_embs):
                        for emb in multi_embs:
                            emb = int(emb.item())
                            if is_QR:
                                (q_addr, r_addr), (q_cmd, r_cmd) = addr_mapper.physical_translation(table, emb)
                                device = "HBM"
                                if cpu_baseline:
                                    for m in range(total_burst):
                                        hit = cache.access(q_addr + 64*m, 'q')
                                        if not hit:
                                            write_trace_line(wf, device, q_addr, "RD", 1)                                

                                    for m in range(total_burst):
                                        hit = cache.access(r_addr + 64*m, 'r')
                                        if not hit:
                                            write_trace_line(wf, device, r_addr, "RD", 1)                                
                                else:
                                    write_trace_line(wf, device, q_addr, q_cmd, total_burst)                                
                                    if not using_prefetch:
                                        if using_subtable_mapping:
                                            write_trace_line(wf, device, r_addr, r_cmd, total_burst)
                                        else:
                                            if r_cmd == "RDWR":
                                                data_move(wf, device, addr_mapper, r_addr, q_addr, pim_level, total_burst)
                                            else:
                                                write_trace_line(wf, device, r_addr, r_cmd, total_burst)

                            elif is_TT_Rec:
                                total_access = addr_mapper.physical_translation(table, emb)
                                device = "HBM"
                                for (a, b, c), (first_cmd, second_cmd, third_cmd) in total_access:
                                    if cpu_baseline:
                                        for m in range(tt_rec_burst):
                                            hit = cache.access(a + 64*m, 'q')
                                            if not hit:
                                                write_trace_line(wf, device, a, "RD", tt_rec_burst)    

                                        for m in range(tt_rec_burst):
                                            hit = cache.access(b + 64*m, 'q')
                                            if not hit:
                                                write_trace_line(wf, device, b, "RD", tt_rec_burst)    

                                        for m in range(tt_rec_burst):
                                            hit = hit and cache.access(c + 64*m, 'q')
                                            write_trace_line(wf, device, c, "RD", tt_rec_burst)    

                                    else:
                                        if not using_prefetch:
                                            if using_subtable_mapping:
                                                write_trace_line(wf, device, a, first_cmd, tt_rec_burst)
                                            else:
                                                if first_cmd == "RDWR":
                                                    data_move(wf, device, addr_mapper, a, b, pim_level, tt_rec_burst)
                                                else:
                                                    write_trace_line(wf, device, a, "RD", tt_rec_burst)

                                        write_trace_line(wf, device, b, "RD", tt_rec_burst)

                                        if using_subtable_mapping:
                                            write_trace_line(wf, device, c, third_cmd, tt_rec_burst)
                                        else:
                                            if third_cmd == "RDWR":
                                                data_move(wf, device, addr_mapper, c, b, pim_level, tt_rec_burst)
                                            else:
                                                write_trace_line(wf, device, c, "RD", tt_rec_burst)

                            else: 
                                total_burst = vec_size // default_vec_size
                                addr = addr_mapper.physical_translation(table, emb)
                                device = "HBM"
                                write_trace_line(wf, device, addr, "RD", total_burst)

                    wf.write('\n')

if __name__ == "__main__":

    addr_map = {
        "rank"      : 0,
        "row"       : 14,
        "bank"      : 2,
        "column"    : 5,
        "bankgroup" : 2,
        "channel"   : 3
    }

    vec_sizes = [128, 256, 512]
    batch_sizes = [4]
    collision = 8
    tt_rank = 16
    dataset ='kaggle'
    pim_level = "bankgroup"

    cpu_baseline = False
    MB_size = 2**20
    CACHE_SIZE = 32*MB_size  # in bytes
    BLOCK_SIZE = 64    # in bytes
    ASSOCIATIVITY = 4
    cache = CacheSimulator.Cache(CACHE_SIZE, BLOCK_SIZE, ASSOCIATIVITY)

    using_subtable_mapping = True
    table_prefetch = False
    all_prefetch = False

    train_data = load_criteo_train_data('./savedata', dataset=dataset)

    train_data_path = './savedata'
    embedding_profile_savefile = f'./savedata/{dataset}/profile.pickle'
    embedding_profiles = load_profile_result(train_data_path, dataset)

    for batch in batch_sizes:
        for vec_size in vec_sizes:
            print(f"Processing vec_size={vec_size}")
            write_trace_file(
                    embedding_profile_savefile=embedding_profile_savefile,
                    embedding_profiles=embedding_profiles,
                    train_data=train_data,
                    called_inside_DLRM=False,
                    dataset=dataset,
                    merge_kaggle_and_terabyte=False,
                    kaggle_duplicate_on_merge=0,
                    total_trace=150,
                    collisions=collision,
                    tt_rank=tt_rank,
                    vec_size=vec_size,
                    batch_size=batch,
                    using_vp=False,
                    table_prefetch=table_prefetch,
                    all_prefetch=all_prefetch,
                    using_subtable_mapping=using_subtable_mapping,
                    addr_map=addr_map,
                    cpu_baseline=cpu_baseline,
                    cache=cache,
                    pim_level=pim_level
                )