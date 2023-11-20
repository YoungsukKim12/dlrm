import numpy as np
import pickle
import sys
import os
from abc import *
from tqdm import tqdm
class TRiM_Address_mapping():
    def __init__(self, k_bits, collisions):
        self.embedding_profiles = self.load_trace_file(savefile='./savedata/profile_collision_{}.pickle'.format(collisions))
        self.collisions = collisions
        self.vec_size = 64
        self.addr_bits = {"rank":1, "row":14, "high_col":5, "bank_group":3, "k":k_bits}
        self.bankgroup_size = 2**k_bits
        self.cache_ratio = 0.5
        self.cache_size = int(self.bankgroup_size * self.cache_ratio)
        self.r_vec_len = [collisions if len(q_tables) > 1 else len(np.nonzero(q_tables)[0]) for q_tables in self.embedding_profiles]
        self.cache_vec_num  =  int(self.cache_size/ self.vec_size) 
        self.hot_vec_loc = self.hot_table_idx()
        if not os.path.exists("profile_collisions_{}_kbits_{}".format(self.collisions, k_bits)):
            self.vec_addr = self.TRiM_address_mapping()
            np.save("profile_collisions_{}_kbits_{}".format(self.collisions, self.cache_size, k_bits), self.vec_addr)
    def hot_table_idx(self):
        cache_vec_num = self.cache_vec_num 
        q_access = [np.sum(q_tables, axis=1).tolist() for q_tables in self.embedding_profiles]
        q_idx = np.array([[i, vec]  for i, table in enumerate(self.embedding_profiles) for vec in range(len(table))])
        q_argsort = np.argsort(np.concatenate(q_access))[::-1]
        self.hot_len = min(cache_vec_num, sum(self.r_vec_len) + len(q_argsort))
        print(cache_vec_num)
        print(sum(self.r_vec_len))
        print(len(q_argsort))
        if cache_vec_num - sum(self.r_vec_len) > 0:
            q_hot_idx = q_idx[q_argsort[:min((cache_vec_num - sum(self.r_vec_len), len(q_argsort)))]]
            q_sort_table = q_hot_idx[np.argsort(q_hot_idx[:,0])]
            uni, cnt = np.unique(q_sort_table[:,0], return_counts=True)
            cum_cnt = np.cumsum(cnt)
            hot_result = []
            hot_t = 0
            for i in range(len(self.embedding_profiles)):
                if i in uni:
                    hot_result.append(q_sort_table[cum_cnt[hot_t-1] if hot_t>=1 else 0: cum_cnt[hot_t], 1].tolist())
                    hot_t += 1
                else:
                    hot_result.append([None])
            
            return hot_result
        else:
            return [[] for _ in range(len(self.embedding_profiles))]

    def load_trace_file(self, savefile='./profile.pickle'):
        if not os.path.exists(savefile):
            print('please run dlrm first!')
            sys.exit()
        else:
            with open(savefile, 'rb') as wf:
                return pickle.load(wf)
    def write_trace_file(self, savefile= "./test_profile_collisions_{}_kbits_{}"):
        savefile = savefile.format(self.collisions, self.addr_bits["k"])
        if not os.path.exists(savefile):
            with open(savefile, 'w') as wf:
                for w in self.vec_addr:
                    wf.write(" ".join(w)+"\n")

               
    def TRiM_address_mapping(self):
        max_k_bits = 2**self.addr_bits["k"]

        cold_addr_acc = 0
        hot_addr_acc = sum(self.r_vec_len) * self.vec_size
        bg_cnt = 0
        print(self.hot_len)
        check_idx = []
        rank_bits = 2**(sum(list(self.addr_bits.values()))- self.addr_bits["rank"])
        vec_addr = [["hot","{}".format(self.vec_size*(sum(self.r_vec_len[:i])+vec)), "r","{}".format(i), "{}".format(vec)]  for i in range(len(self.embedding_profiles)) for vec in range(self.r_vec_len[i])]
        for i, emb_table in enumerate(self.embedding_profiles):
            for vec in tqdm(range(len(emb_table))):
                if vec in self.hot_vec_loc[i]:
                    vec_addr.append(["hot", "{}".format(hot_addr_acc), "q","{}".format(i),"{}".format(vec)])
                    hot_addr_acc += self.vec_size
                else:
                    if cold_addr_acc % max_k_bits == 0:
                        if bg_cnt < 2**self.addr_bits["bank_group"]:
                            cold_addr_acc += self.vec_size * self.hot_len
                            bg_cnt += 1
                        elif cold_addr_acc % rank_bits == 0:
                            bg_cnt = 1
                            cold_addr_acc += self.vec_size * self.hot_len
                    else:
                        cold_addr_acc += self.vec_size
                        #print(cold_addr_acc)
                    vec_addr.append(["cold", "{}".format(cold_addr_acc), "q","{}".format(i),"{}".format(vec)])
        return vec_addr
TRiM_mapping = TRiM_Address_mapping(k_bits=16, collisions=16)
TRiM_mapping.write_trace_file()  