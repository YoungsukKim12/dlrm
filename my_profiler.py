import numpy as np

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

def write_hot_q_profile_result(f, hot_q_idx, hot_q_hit_ratio, r_in_q):
    '''
        Writes the followings:
            1. hot q idx and its access ratio in its category table
            2. r access ratio of hot qs
            3. r access ratio in its category table
    '''
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
    f.write('total q # :  %d\n\n' %MyProfiler.table_profiles[table_id].shape[0])

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

def process_profile_data(prof_per_table, hot_q_ratio, cold_thresh=0.00001):
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
    frozen_q = q_sum[q_sum == 0].shape[0]

    sort_idx = np.argsort(-q_sum)
    hot_q_idx = index_array[sort_idx][:top_k_q]
    # cold_q_idx = np.argsort(q_sum)

    hot_q_hit_ratio = q_sum[sort_idx][:top_k_q] / total_access
    # cold_q_hit_ratio = q_sum[cold_q_idx] / total_access
    r_in_q = []
    for j, r in enumerate(prof_per_table):
        r_in_q.append(r)

    return hot_q_idx, hot_q_hit_ratio, frozen_q, r_in_q, total_access

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
            hot_q_idx, hot_q_hit_ratio, frozen_q, r_in_q, total_access = process_profile_data(prof_per_table, hot_q_ratio)
            write_table_info(f, i)
            write_hot_q_profile_result(f, hot_q_idx, hot_q_hit_ratio, r_in_q)
            write_frozen_q_profile_result(f, frozen_q)
            total_access_list.append(total_access)
        write_table_profile_result(f, total_access_list)            