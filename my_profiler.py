import numpy as np
import math
import random
import pickle
import os
import dlrm_data_pytorch as dp
import sys
from address_mapping import AddressMapping

class MyProfiler:

    table_profiles = []

    @staticmethod
    def set_qr_profile(q_len, r_len):
        qr_profiles = np.zeros((q_len,r_len))
        MyProfiler.table_profiles.append(qr_profiles)

    @staticmethod
    def record_qr_profile(i, q, r):
        q = q.detach().cpu().numpy()
        r = r.detach().cpu().numpy()
        MyProfiler.table_profiles[i][q,r] += 1

def write_frozen_q_profile_result(f, frozen_q):
    '''
        Writes the followings:
            1. zero access q total number
    '''

    # cold_q_idx = cold_q_idx[cold_q_hit_ratio < 0.0001]
    # zero_q_idx = cold_q_idx[cold_q_hit_ratio == 0]

    f.write('cold q vectors:\n')
    # f.write('total cold q vectors : %d\n' %cold_q_idx.shape[0])
    f.write('total 0-access q vectors : %d' %frozen_q)
    f.write('\n\n')

def write_hot_q_profile_result(f, hot_q_idx, hot_q_hit_ratio, r_in_q, hot_vectors):
    '''
        Writes the followings:
            1. hot q idx and its access ratio in its category table
            2. r access ratio of hot qs
            3. r access ratio in its category table
    '''

    f.write('total hot vectors : %d \n\n' %hot_vectors)
    # write hot q idx and access ratio
    for j, idx in enumerate(hot_q_idx):
        if hot_q_hit_ratio[j] >= 0.005:
            f.write('%d idx : %1.3f   ' % (idx, hot_q_hit_ratio[j]))
            if (j+1) % 3 == 0:
                f.write('\n')
    f.write('\n                            <<r in top q>>\n')

    # write r access ratio of hot qs / total q
    if hot_q_idx.shape[0] > 0:
        r_in_q = np.array(r_in_q)
        top_q_r_sum = np.sum(r_in_q[hot_q_idx], axis=0)
        total_r_sum = np.sum(r_in_q, axis=0)
        f.write('<r ratio at hot qs> : %s  \n' %(str(top_q_r_sum/np.sum(top_q_r_sum))))
        f.write('<r ratio at total> :  %s \n' %(str(total_r_sum/np.sum(total_r_sum))))
    f.write('\n')

def write_table_info(f, table_id):
    '''
        Writes the followings:
            1. total number of q's in each category table
    '''
    # write the number of total q in each table
    f.write('                          <<At table %d>>\n\n' %table_id)
    f.write('total q # :  %d\n' %MyProfiler.table_profiles[table_id].shape[0])

def write_table_profile_result(f, total_access_list, top_k=-1):
    '''
        Writes the followings:
            1. category tables sorted with their access ratio
            2. each table's access ratio
    '''

    total_access_list = np.array(total_access_list)
    access_tops = np.argsort(-total_access_list)[:top_k]
    access_tops_ratio = total_access_list[access_tops] / (306969*128)
    f.write('hot tables : ' + str(access_tops)+'\n')
    f.write('ratio : ' + str(access_tops_ratio))

def process_profile_data(prof_per_table, hot_q_ratio, cold_thresh=0.00001, hot_access_ratio=0.8):
    '''
        Returns the followings:
            1. hot q vectors' idx and its access ratio
            2. each table's access ratio
            3. r access ratio of each hot q's
            4. total accesses in each category table
    '''

    top_k_q = int(prof_per_table.shape[0] * hot_q_ratio)
    top_k_r = int(prof_per_table.shape[1])
    total_access = np.sum(prof_per_table)

    index_array = np.array(list(range(prof_per_table.shape[0])))
    q_sum = np.sum(prof_per_table, axis=1)
    sort_idx = np.argsort(-q_sum)
    hot_q_idx = index_array[sort_idx][:top_k_q]
    frozen_q = q_sum[q_sum == 0].shape[0]

    hot_access = 0
    hot_vectors = 0
    for i, access in enumerate(q_sum[np.argsort(-q_sum)]):
        hot_access += access
        if hot_access >= total_access*hot_access_ratio:
            hot_vectors = i+1
            break

    hot_q_hit_ratio = q_sum[sort_idx][:top_k_q] / total_access
    # cold_q_idx = np.argsort(q_sum)
    # cold_q_hit_ratio = q_sum[cold_q_idx] / total_access
    r_in_q = []
    for j, r in enumerate(prof_per_table):
        r_in_q.append(r)

    return hot_q_idx, hot_q_hit_ratio, frozen_q, r_in_q, hot_vectors, total_access

def total_compressed_embs():
    tot_embs = 0

    for table in MyProfiler.table_profiles:
        tot_embs += table.shape[0]
        tot_embs += table.shape[1]
    return tot_embs

def write_profile_result(collisions, hot_q_ratio=0.01):
    fname = './qr_profiles_%d.txt' %collisions
    with open(fname ,'w') as f:
        total_access_list = []
        for i, prof_per_table in enumerate(MyProfiler.table_profiles):
            hot_q_idx, hot_q_hit_ratio, frozen_q, r_in_q, hot_vectors, total_access = process_profile_data(prof_per_table, hot_q_ratio)
            write_table_info(f, i)
            write_hot_q_profile_result(f, hot_q_idx, hot_q_hit_ratio, r_in_q, hot_vectors)
            write_frozen_q_profile_result(f, frozen_q)
            total_access_list.append(total_access)
        write_table_profile_result(f, total_access_list)            


def load_profile_result(called_inside_dlrm, savefile):
    profiles = None
    if called_inside_dlrm:
        profiles = MyProfiler.table_profiles
        with open(savefile, 'wb') as sf:
            pickle.dump(profiles, sf)
    else:
        if not os.path.exists(savefile):
            print('please run dlrm first!')
            sys.exit()
        else:
            with open(savefile, 'rb') as wf:
                profiles = pickle.load(wf)            

    return profiles

def get_physical_address(addr_mapper, using_bg_map, index, emb_value, collisions=None):
    if using_bg_map:
        in_HBM, emb_physical_addr = addr_mapper.bankgroup_based_physical_address_mapping(index, emb_value)
    else:
        in_HBM, emb_physical_addr = addr_mapper.basic_physical_address_mapping(index, emb_value, collisions)

    return in_HBM, emb_physical_addr

def write_trace_file(
        called_inside_dlrm=True,
        collisions=4, 
        readfile='./input/train.txt',
        savefile='./profile.pickle', 
        train_data=None, 
        qr_flag=False, 
        using_bg_map=False
    ):

    profiles = load_profile_result(called_inside_dlrm, savefile)
    addr_mapper = AddressMapping(profiles, using_bg_map=using_bg_map)
    total_data = len(train_data)

    if using_bg_map:
        writefile = './bg_map_trace_col_%d.txt' % collisions
    else:
        writefile = './random_trace_col_%d.txt' % collisions

    print("writing trace file...")
    with open(writefile, 'w') as wf:
        for i, data in enumerate(train_data):

            if i % 1024 == 0:
                print(f"{i}/{total_data} processed")
            if i > 20000:
                print('done writing tracing file')
                break

            dense, feat, y = data
            for i, emb in enumerate(feat):
                q_emb = emb // collisions

                # write q vector
                in_HBM, emb_physical_addr = get_physical_address(addr_mapper, using_bg_map, i, q_emb)
                device = "HBM" if in_HBM else "DIMM"
                wf.write(f"{device} {emb_physical_addr}\n")

                # write r vector                
                if qr_flag and not using_bg_map:
                    r_emb = emb % collisions
                    in_HBM, emb_physical_addr = get_physical_address(addr_mapper, using_bg_map, i, r_emb, collisions)

                    device = "HBM" if in_HBM else "DIMM"
                    wf.write(f"{device} {emb_physical_addr}\n")
            wf.write('\n')


if __name__ == "__main__":
    profiles = None
    savefile = './profile.pickle'
    collision = 4
    print('loading train data...')

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
    print('train data load complete!')

    write_trace_file(False, collision, train_data=train_data)