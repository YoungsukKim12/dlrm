import numpy as np
import pickle
import sys
import os
class TRiM_Address_mapping():
    def __init__(self, k_bits, collisions, vec_size):
        self.embedding_profiles = self.load_trace_file(savefile='./savedata/profile_collision_{}.pickle'.format(collisions))
        self.collisions = collisions
        self.vec_size = vec_size
        self.addr_bits = {"rank":1, "row":14, "high_col":5, "bank_group":3, "k":k_bits}
        self.bankgroup_size = 2**k_bits
        self.cache_ratio = 0.5
        self.cache_size = int(self.bankgroup_size * self.cache_ratio)
        self.r_vec_len = [collisions if len(q_tables) > 1 else len(np.nonzero(q_tables)[0]) for q_tables in self.embedding_profiles]
        self.cache_vec_num  =  int(self.cache_size/ self.vec_size)
        self.hot_vec_loc = self.hot_table_idx()
        self.vec_addr = self.TRiM_address_mapping()
        if not os.path.exists("profile_collisions_{}_kbits_{}.npy".format(self.collisions, k_bits)):
            np.save("profile_collisions_{}_kbits_{}".format(self.collisions, k_bits), self.vec_addr)
            self.write_trace_file()

    def hot_table_idx(self):
        cache_vec_num = self.cache_vec_num 
        q_access = [np.sum(q_tables, axis=1).tolist() for q_tables in self.embedding_profiles]
        q_idx = np.array([[i, vec]  for i, table in enumerate(self.embedding_profiles) for vec in range(len(table))])
        q_argsort = np.argsort(np.concatenate(q_access))[::-1]
        self.hot_len = min(cache_vec_num, sum(self.r_vec_len) + len(q_argsort))
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
    def write_trace_file(self, savefile= "./test_profile_collisions_{}_kbits_{}_vec_size_{}", data=None):
        if savefile == "./test_profile_collisions_{}_kbits_{}_vec_size_{}":
            savefile = savefile.format(self.collisions, self.addr_bits["k"], self.vec_size)
            data = self.vec_addr
        if data is not None:
            with open(savefile, 'w') as wf:
                for w in data:
                    wf.write(" ".join(w)+"\n")

               
    def TRiM_address_mapping(self):
        max_k_bits = 2**self.addr_bits["k"]

        cold_addr_acc = 0
        hot_addr_acc = sum(self.r_vec_len) * self.vec_size
        bg_cnt = 0
        
        self.q_tables_pa = [[] for _ in range(len(self.embedding_profiles))]
        self.r_tables_pa = [[] for _ in range(len(self.embedding_profiles))]
        for i in range(len(self.embedding_profiles)):
            for vec in range(self.r_vec_len[i]):
                self.r_tables_pa[i].append(self.vec_size*(sum(self.r_vec_len[:i])+vec))

        rank_bits = 2**(sum(list(self.addr_bits.values()))- self.addr_bits["rank"])
        vec_addr = [["hot","{}".format(self.vec_size*(sum(self.r_vec_len[:i])+vec)), "r","{}".format(i), "{}".format(vec)]  for i in range(len(self.embedding_profiles)) for vec in range(self.r_vec_len[i])]
        for i, emb_table in enumerate(self.embedding_profiles):
            for vec in range(len(emb_table)):
                if vec in self.hot_vec_loc[i]:
                    vec_addr.append(["hot","{}".format(hot_addr_acc), "q","{}".format(i),"{}".format(vec)])
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
                    else:
                        cold_addr_acc += self.vec_size
                    vec_addr.append(["cold","{}".format(cold_addr_acc), "q","{}".format(i),"{}".format(vec)])
                self.q_tables_pa[i].append(int(vec_addr[-1][1]))
        return vec_addr


    
class CacheBlock:
    def __init__(self, tag):
        self.tag = tag
        self.lru_counter = 0  # For LRU policy

class CacheSet:
    def __init__(self, associativity):
        self.blocks = [None] * associativity

    def access_block(self, tag):
        # Check for hit and update LRU counter
        for block in self.blocks:
            if block and block.tag == tag:
                block.lru_counter = 0
                return True  # Cache hit

        # Cache miss, load the block
        self.load_block(tag)
        return False

    def load_block(self, tag):
        # Find the least recently used block or an empty block
        lru_block = max(self.blocks, key=lambda b: b.lru_counter if b else -1)
        if lru_block:
            lru_block.tag = tag
            lru_block.lru_counter = 0
        else:
            # If there's an empty slot, use it
            for i in range(len(self.blocks)):
                if self.blocks[i] is None:
                    self.blocks[i] = CacheBlock(tag)
                    return

        # Update LRU counters
        for block in self.blocks:
            if block:
                block.lru_counter += 1


class Cache: 
    def __init__(self, size, block_size, associativity):
        self.num_sets = size // (block_size * associativity)
        #associativity = int(size//block_size) #fully
        self.sets = [CacheSet(associativity) for _ in range(self.num_sets)]
        self.block_size = block_size
        self.associativity = associativity
        self.hits = {'q': 0, 'r': 0}
        self.misses = {'q': 0, 'r': 0}

    def access(self, address, category):
        tag = address // (self.block_size * self.num_sets)
        set_index = (address // self.block_size) % self.num_sets

        if self.sets[set_index].access_block(tag):
            self.hits[category] += 1
        else:
            self.misses[category] += 1

    def category_hit_rate(self, category):
        total_accesses = sum(self.hits.values()) + sum(self.misses.values())
        return self.hits[category] / total_accesses if total_accesses > 0 else 0

    def overall_hit_rate(self):
        total_hits = sum(self.hits.values())
        total_misses = sum(self.misses.values())
        total_accesses = total_hits + total_misses
        return total_hits / total_accesses if total_accesses > 0 else 0

def pa2blockadrr( BLOCK_SIZE, VECTOR_SIZE, pa):
    addr = []
    for i in range(int(np.ceil(VECTOR_SIZE/BLOCK_SIZE))):
        addr.append(pa + i*BLOCK_SIZE)
    return addr

def query2pa(q_tables_pa_l, r_tables_pa_l, collisions ,cache_l, cache_size_l, block_size_l, vec_sizes, ps_data_pt = "/mnt/storage/sjm/kys_dlrm/dlrm/input/"):
    data = None
    if True:
        f = np.load(ps_data_pt + "kaggleAdDisplayChallenge_processed.npz")
        data = np.transpose(f["X_cat"])
    for t_num, k in enumerate(data):
        for j in k:
            q_idx, r_idx = np.divmod(j, collisions)
            for c_num in range(len(cache_l)):
                for b_num, block_size in enumerate(block_size_l):
                    for vec_idx, vec_size in enumerate(vec_sizes):
                        q_pa = int(q_tables_pa_l[vec_idx][t_num][int(q_idx)])
                        r_pa = int(r_tables_pa_l[vec_idx][t_num][int(r_idx)])
                        for addr in pa2blockadrr(block_size, vec_size, q_pa):
                            cache_l[c_num][b_num][vec_idx].access(addr, "q")
                        for addr in pa2blockadrr(block_size, vec_size, r_pa):
                            cache_l[c_num][b_num][vec_idx].access(addr, "r")
        result_l = []
        print("\nTable: {}".format(t_num+1))
        for i, c_size in enumerate(cache_size_l):
            print("Cache Size: {}MB".format(c_size))
            for r, b_size in enumerate(block_size_l):
                print("Block: {}B".format(b_size))
                for k, v_size in enumerate(vec_sizes):
                    print("Vector Size: {}B".format(v_size))
                    print(f"Overall Hit Rate: {cache_l[i][r][k].overall_hit_rate():.2%}")
                    print(f"Hit Rate of Category 'q' Among All Categories: {cache_l[i][r][k].category_hit_rate('q'):.2%}")
                    print(f"Hit Rate of Category 'r' Among All Categories: {cache_l[i][r][k].category_hit_rate('r'):.2%}")
                    result_l.append([cache_l[i][r][k].overall_hit_rate(), cache_l[i][r][k].category_hit_rate('q'), cache_l[i][r][k].category_hit_rate('r')])
        np.save("Cache_simulation_total", result_l)
    return cache_l




collisions = 4
k_bits = 18
result = []
MB = 2**20
ASSOCIATIVITY = 4  # 4-way set associative

cache_size_l = [1,2,4,8]
block_size_l = [64]#[64,128,256,512]
vec_sizes = [64,128,256,512]
cache_l = [[[Cache(cache_size*MB, block_size, ASSOCIATIVITY) for _ in range(len(vec_sizes))] for block_size in block_size_l] for cache_size in cache_size_l]
"""
CACHE_SIZE = 1*MB  # in bytes
BLOCK_SIZE = 64    # in bytes
cache = Cache(CACHE_SIZE, BLOCK_SIZE, ASSOCIATIVITY)
"""
q_tables_pa_l = []
r_tables_pa_l = []
# Cache parameters (example values)
for vector_size in vec_sizes:
    TRiM_mapping = TRiM_Address_mapping(k_bits=k_bits, collisions=collisions, vec_size=vector_size)
    # Initialize cache 
    q_tables_pa_l.append(TRiM_mapping.q_tables_pa)
    r_tables_pa_l.append(TRiM_mapping.r_tables_pa)
    
result = query2pa(q_tables_pa_l, r_tables_pa_l, collisions ,cache_l, cache_size_l, block_size_l, vec_sizes)
result_l = []
print()
for i, c_size in enumerate(cache_size_l):
    print("Cache Size: {}MB".format(c_size))
    for j, b_size in enumerate(block_size_l):
        print("Block: {}B".format(b_size))
        for k, v_size in enumerate(vec_sizes):
            print("Vector Size: {}B".format(v_size))
            print(f"Overall Hit Rate: {result[i][j][k].overall_hit_rate():.2%}")
            print(f"Hit Rate of Category 'q' Among All Categories: {result[i][j][k].category_hit_rate('q'):.2%}")
            print(f"Hit Rate of Category 'r' Among All Categories: {result[i][j][k].category_hit_rate('r'):.2%}")
            result_l.append([result[i][j][k].overall_hit_rate(), result[i][j][k].category_hit_rate('q'), result[i][j][k].category_hit_rate('r')])
np.save("Cache_simulation_total", result_l)