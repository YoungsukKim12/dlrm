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
from itertools import chain
import argparse

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
    print(f"loading embedding profile from {savefile}")
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
        if physical_addr != -1: # cached or skipped access
            wf.write(f"{device} {command} {physical_addr} {int(total_burst)} \n")

def data_move(write_file, device, addr_mapper, addr, cmp_addr, pim_level, burst):
    write_trace_line(write_file, device, addr, "RD", burst)
    transfer_addr, same_node = addr_mapper.map_to_same_node(pim_level, cmp_addr, addr, randomize_row=True)
    if not same_node:
        write_trace_line(write_file, device, transfer_addr, "DR", burst)

def write_trace_file(
        emb_prof_path='./savedata',
        train_data=None,
        dataset='kaggle',
        total_trace=10,
        batch_size=4,
        collisions=4,
        tt_rank=16, 
        vec_size=64,
        cpu_baseline=False,
        cache=None,
        table_prefetch=True,
        all_prefetch=False,
        using_subtable_mapping=True,
        addr_map={},
        addr_mappers=[],
        pim_level="bankgroup"
    ):

    if len(addr_map) != 6:
        print("please provide correct address mapping!")
        sys.exit()

    total_batch = total_trace // batch_size
    embedding_profiles = load_profile_result(emb_prof_path, dataset)

    print('save & load complete')

    default_vec_size = 64
    total_burst = vec_size // default_vec_size
    tt_rec_burst = tt_rank * 4 // default_vec_size
    total_data = len(train_data)

    multi_hot = Multihot(
                    multi_hot_sizes=[80 for i in range(len(embedding_profiles))],
                    num_embeddings_per_feature=[len(table) for table in embedding_profiles],
                    batch_size=1,
                    collect_freqs_stats=False,
                    dist_type='pareto',
                    dataset=dataset
                )

    for addr_mapper in addr_mappers:
        mapper_name = addr_mapper.mapper_name()
        using_prefetch = all_prefetch or table_prefetch and "ProactivePIM" in mapper_name
        is_QR = True if "QR" in mapper_name else False
        is_TT_Rec = True if "TT" in mapper_name else False
        
        wfile_head = f'./{mapper_name}_{dataset}_{vec_size}B_'        
        if cpu_baseline:
            wfile = wfile_head + f'baseline.txt'
        elif using_subtable_mapping:
            if table_prefetch:
                wfile = wfile_head + f'tw_pf.txt'
            elif all_prefetch:
                wfile = wfile_head + f'all_pf.txt'
            else:
                wfile = wfile_head + f'submap.txt'
        print(f"write file name : {wfile}")

        with open(wfile, 'w') as wf:
            total_data_move = 0
            for i in range(len(train_data)//batch_size):
                batch_data = [feat for _, feat, _ in train_data[i*batch_size:(i+1)*batch_size]]
                batch_data = np.array(batch_data, dtype=np.int64)
                batch_data = np.transpose(batch_data)
                multi_hot_indices = multi_hot.make_new_batch(lS_i=batch_data, batch_size=batch_size)
                multi_hot_indices = np.transpose(multi_hot_indices)

                if i % 10 == 0:
                    print(f"{i}/{total_batch} trace processed")
                if i > total_batch:
                    break

                # prefetch only once if all_prefetch flag is set
                if all_prefetch and i == 0:
                    for table in range(len(embedding_profiles)):
                        prefetch_addrs = addr_mapper.get_prefetch_physical_address(table)
                        burst = tt_rec_burst if is_TT_Rec else total_burst
                        write_trace_line(wf, device, prefetch_addrs[i], "PR", burst)

                for table, batch_embs in enumerate(multi_hot_indices):                 

                    # write transfer/prefetch command for each table
                    # inital execution has no result to transfer // batch_size * 4 => four bank groups
                    transfers = 4*batch_size if not (i == 0 and table == 0) else 0 
                    if using_prefetch:
                        prefetch_addrs = addr_mapper.get_prefetch_physical_address(table)
                        overhead = max(transfers, len(prefetch_addrs))
                        for i in range(overhead):
                            device = "HBM"
                            if i < transfers:
                                write_trace_line(wf, device, 0, "TR", total_burst)
                            if i < len(prefetch_addrs):
                                burst = tt_rec_burst if is_TT_Rec else total_burst
                                write_trace_line(wf, device, prefetch_addrs[i], "PR", burst)
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
                                    q_hit = True
                                    r_hit = True
                                    
                                    for m in range(total_burst):
                                        q_hit = q_hit and cache.access(q_addr + 64*m, 'q')                           
                                        r_hit = r_hit and cache.access(r_addr + 64*m, 'r')
                                        
                                    if not q_hit:
                                        write_trace_line(wf, device, q_addr, "RD", 1)
                                    if not r_hit:
                                        write_trace_line(wf, device, r_addr, "RD", 1)                                
                                else:
                                    write_trace_line(wf, device, q_addr, q_cmd, total_burst)                                
                                    if not using_prefetch:
                                        if using_subtable_mapping:
                                            write_trace_line(wf, device, r_addr, r_cmd, total_burst)
                                        else:
                                            if r_cmd == "RDWR":
                                                data_move(wf, device, addr_mapper, r_addr, q_addr, pim_level, total_burst)
                                                total_data_move += 1
                                            else:
                                                write_trace_line(wf, device, r_addr, r_cmd, total_burst)

                            elif is_TT_Rec:
                                total_access = addr_mapper.physical_translation(table, emb)
                                device = "HBM"
                                for (a, b, c), (first_cmd, second_cmd, third_cmd) in total_access:
                                    if cpu_baseline:
                                        a_hit = True
                                        b_hit = True
                                        c_hit = True
                                        for m in range(tt_rec_burst):
                                            a_hit = a_hit and cache.access(a + 64*m, 'q')
                                            c_hit = c_hit and cache.access(c+64*m, 'q')
                                        for m in range(math.pow(tt_rec_burst, 2)):
                                            b_hit = b_hit and cache.access(b + 64*m, 'q')

                                        if not a_hit:
                                            write_trace_line(wf, device, a, "RD", tt_rec_burst)    
                                        if not b_hit:
                                            write_trace_line(wf, device, b, "RD", math.pow(tt_rec_burst,2)) 
                                        if not c_hit:
                                            write_trace_line(wf, device, c, "RD", tt_rec_burst)    
                                              
                                    else:
                                        if not using_prefetch:
                                            if using_subtable_mapping:
                                                write_trace_line(wf, device, a, first_cmd, tt_rec_burst)
                                                write_trace_line(wf, device, c, third_cmd, tt_rec_burst)
                                            else:
                                                if first_cmd == "RDWR":
                                                    total_data_move += 1
                                                    data_move(wf, device, addr_mapper, a, b, pim_level, tt_rec_burst)
                                                else:
                                                    write_trace_line(wf, device, a, "RD", tt_rec_burst)

                                                if third_cmd == "RDWR":
                                                    total_data_move += 1
                                                    data_move(wf, device, addr_mapper, c, b, pim_level, tt_rec_burst)
                                                else:
                                                    write_trace_line(wf, device, c, "RD", tt_rec_burst)

                                        write_trace_line(wf, device, b, "RD", math.pow(tt_rec_burst,2))

                            else: 
                                total_burst = vec_size // default_vec_size
                                addr = addr_mapper.physical_translation(table, emb)
                                device = "HBM"
                                write_trace_line(wf, device, addr, "RD", total_burst)

                    wf.write('\n')
            print("total_data_move : ", total_data_move)

def addrmap_generator(
        mapper_name='ProactivePIM',
        embedding_profiles=None,
        qr=True,
        tt_rec=False,
        vec_size=128, 
        collisions=4, 
        rank=16, 
        using_prefetch=False, 
        using_mapping=False, 
        addr_map={}, 
        pim_level='bankgroup', 
    ):
    
    addr_mappers = []
    
    if qr:
        addr_mappers.append(
                ProactivePIMTranslation(
                    embedding_profiles=embedding_profiles, 
                    vec_size=vec_size, 
                    HBM_size_gb=4, 
                    is_QR=True, 
                    collisions=collisions, 
                    using_prefetch=using_prefetch,
                    using_subtable_mapping=using_subtable_mapping,
                    addr_map=addr_map,
                    pim_level=pim_level,
                    mapper_name=mapper_name+"QR"
                )
        )

    if tt_rec:
        addr_mappers.append(
                ProactivePIMTranslation(
                    embedding_profiles=embedding_profiles, 
                    vec_size=vec_size, 
                    HBM_size_gb=4, 
                    is_TT_Rec=True, 
                    tt_rank=tt_rank, 
                    using_prefetch=using_prefetch,
                    using_subtable_mapping=using_subtable_mapping,
                    using_gemv_dist=True,
                    addr_map=addr_map,
                    pim_level=pim_level,
                    mapper_name=mapper_name+"TT"
                )
        )

    return addr_mappers


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=string, default="kaggle", help="dataset : kaggle or terabyte")
    parser.add_argument('--qr', type=bool, default=False, help="using qr") 
    parser.add_argument('--tt', type=bool, default=False, help="using tt")
    parser.add_argument('--vec_sizes', type=list, default=[128], help="vector size in list format")
    parser.add_argument('--batch', type=int, default=4, help="batch size")
    parser.add_argument('--collision', type=int, default=4, help="qr collision") 
    parser.add_argument('--tt_rank', type=int, default=16, help="tt rank")
    parser.add_argument('--pim_level', type=string, default='bankgroup', help="pim level : rank, bankgroup, bank")
    parser.add_argument('--baseline', type=bool, default=False, help="baseline trace")
    parser.add_argument('--submap', type=bool, default=False, help="subtable mapping trace")
    parser.add_argument('--allprefetch', type=bool, default=False, help="all prfetch trace")
    parser.add_argument('--tableprefetch', type=bool, default=False, help="table prefetch trace")
    parser.parse_args()    

    addr_map = {
        "rank"      : 0,
        "row"       : 14,
        "bank"      : 2,
        "channel"   : 3,
        "bankgroup" : 2,
        "column"    : 5
    }

    vec_sizes = args.vec_sizes
    batch_size = args.batch
    collision = args.collision
    tt_rank = args.rank
    dataset = args.dataset
    pim_level = args.pim_level
    cpu_baseline = args.baseline
    using_subtable_mapping = args.submap
    table_prefetch = args.tableprefetch
    all_prefetch = args.allprefetch
        
    MB_size = 2**20
    CACHE_SIZE = 32*MB_size  # in bytes
    BLOCK_SIZE = 64    # in bytes
    ASSOCIATIVITY = 4
    cache = CacheSimulator.Cache(CACHE_SIZE, BLOCK_SIZE, ASSOCIATIVITY)

    print("Loading embedding profiles for trace generation")

    train_data = load_criteo_train_data('./savedata', dataset=dataset)
    # train_data_path = './savedata'
    # embedding_profile_savefile = f'./savedata/{dataset}/profile.pickle'
    # embedding_profiles = load_profile_result(train_data_path, dataset)
    # embedding_profiles = np.array(embedding_profiles)

    for vec_size in vec_sizes:
        using_prefetch = table_prefetch or all_prefetch
        print("Generating address mappers...")
        ProactivePIM_maps = addrmap_generator(embedding_profiles, vec_size, collision, tt_rank, using_prefetch, using_subtable_mapping, addr_map, "bankgroup", "ProactivePIM")
        # ProactivePIM_maps = addrmap_generator(embedding_profiles, vec_size, collision, tt_rank, using_prefetch, using_subtable_mapping, addr_map, "bankgroup", "ProactivePIM_All_Prefetch")
        # RecNMP_maps = addrmap_generator(embedding_profiles, vec_size, collision, tt_rank, False, using_subtable_mapping, addr_map, "rank", "RecNMP")
        # SPACE_maps = addrmap_generator(embedding_profiles, vec_size, collision, tt_rank, False, using_subtable_mapping, addr_map, "rank", "SPACE")
        # addr_mappers = list(chain(ProactivePIM_maps, RecNMP_maps, SPACE_maps))
        addr_mappers = ProactivePIM_maps
        print("Generating traces for DRAMsim3...")
        
        write_trace_file(
                emb_prof_path='./savedata',
                train_data=train_data,
                dataset=dataset,
                total_trace=150,
                collisions=collision,
                tt_rank=tt_rank,
                vec_size=vec_size,
                batch_size=batch,
                table_prefetch=table_prefetch,
                all_prefetch=all_prefetch,
                using_subtable_mapping=using_subtable_mapping,
                addr_map=addr_map,
                addr_mappers=addr_mappers,
                cpu_baseline=cpu_baseline,
                cache=cache,
                pim_level=pim_level
            )