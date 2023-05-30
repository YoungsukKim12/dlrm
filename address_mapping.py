import math
import numpy as np
import random

class AddressMapping():

    HBM_bit_width_4GB = {'rank' : 0, 'row' : 14, 'colhigh' : 5, 'channel' : 4, 'bankgroup' : 2, 'bank' : 4, 'collow' : 5, 'offset' : 3}

    def __init__(
        self, 
        profiles, 
        HBM_size_gb=4, 
        DIMM_size_gb=8, 
        hot_access_ratio=1,
        vec_size = 64,
        tables_per_bankgroup=3,
        end_iter=20000, 
        using_bg_map=True
    ):

        # basic parameters
        self.profiles = profiles
        self.hot_access_ratio = hot_access_ratio
        self.vec_size = vec_size
        self.page_offset = math.pow(2, 12)
        self.tables_per_bankgroup = 3
        self.collisions = 4

        # DRAM size and ppns
        GB_size = math.pow(2, 30)
        HBM_Size = HBM_size_gb * GB_size
        DRAM_Size = DIMM_size_gb * GB_size
        HBM_max_page_number = int(HBM_Size // self.page_offset)
        DRAM_max_page_number = int(DRAM_Size // self.page_offset)
        HBM_ppn_bits = int(math.log2(HBM_max_page_number))

        # for basic address mapping (use random vpn -> ppn mapping)
        self.HBM_page_translation = [i for i in range(HBM_max_page_number)]
        self.DRAM_page_translation = [i for i in range(DRAM_max_page_number)]
        random.shuffle(self.HBM_page_translation)
        random.shuffle(self.DRAM_page_translation)

        # for bg based address mapping
        self.total_bankgroups = int(math.pow(2, (AddressMapping.HBM_bit_width_4GB['rank'] + AddressMapping.HBM_bit_width_4GB['channel'] + AddressMapping.HBM_bit_width_4GB['bankgroup'])))
        self.bankgroup_size = int(HBM_Size // self.total_bankgroups)
        print("total bank groups : ", self.total_bankgroups)
        print("bankgroup size in bytes : ", self.bankgroup_size)

        # preprocess
        print('preprocess in progress...')
        if not using_bg_map:
            self.hot_vec_loc = self.profile_hot_vec_location(self.profiles, self.hot_access_ratio)
            self.DRAM_table_start_address, self.HBM_table_start_address = self.basic_logical_address_mapping(self.hot_vec_loc, self.profiles, self.vec_size, self.collisions)
        else:
            self.hot_vec_loc = self.profile_hot_vec_location(self.profiles, self.hot_access_ratio)
            self.table_record, self.bg_record = self.bankgroup_based_logical_address_mapping(self.total_bankgroups, self.tables_per_bankgroup, self.hot_vec_loc, self.bankgroup_size, self.vec_size)
            self.DRAM_table_start_address, _ = self.basic_logical_address_mapping(self.hot_vec_loc, self.profiles, self.vec_size, 0)

            self.bg_based_mapping_row = [i for i in range(int(math.pow(2, AddressMapping.HBM_bit_width_4GB['row'])))]
            self.bg_based_mapping_colhigh = [i for i in range(int(math.pow(2, AddressMapping.HBM_bit_width_4GB['colhigh'])))]
            random.shuffle(self.bg_based_mapping_row)
            random.shuffle(self.bg_based_mapping_colhigh)

            print('table_record', self.table_record)
            print('bankgroup_record', self.bg_record)


    def profile_hot_vec_location(self, profiles, hot_access_ratio):
        '''
            returns :  hot vector index list of each table
            
            hot vector index list is sorted for each table
        '''

        hot_access_ratio = self.hot_access_ratio
        total_access = 0
        table_len = len(profiles)
        idx_list = [0 for _ in range(table_len)]
        hot_vec_loc = [[] for _ in range(table_len)]
        sorted_q_sum_tables = []
        sorted_idx_tables = []

        for i, prof_per_table in enumerate(profiles):
            total_access += np.sum(prof_per_table)
            q_sum = np.sum(prof_per_table, axis=1)
            sort_idx = np.argsort(-q_sum)
            sorted_q_sum_tables.append(q_sum[sort_idx]) 
            sorted_idx_tables.append(sort_idx)

        while not hot_access_ratio < 0:
            hot_vecs = [int(qsum_table[idx_list[i]]) if not idx_list[i] == -1 else -10 for i, qsum_table in enumerate(sorted_q_sum_tables)]
            hot_vec_table_idx = np.argmax(hot_vecs)
            hot_vec_loc[hot_vec_table_idx].append(np.where(sorted_idx_tables[hot_vec_table_idx] == idx_list[hot_vec_table_idx])[0][0])
            idx_list[hot_vec_table_idx] += 1
            if len(sorted_q_sum_tables[hot_vec_table_idx]) == idx_list[hot_vec_table_idx]:
                idx_list[hot_vec_table_idx] = -1
            hot_access_ratio -= np.max(hot_vecs) / total_access

        return hot_vec_loc

    def basic_logical_address_mapping(self, hot_vec_loc, profiles, vec_size, collisions):
        DRAM_table_start_address = [0 + vec_size * (prof_per_table.shape[0]-len(hot_vec_loc[i])) for i, prof_per_table in enumerate(profiles)]
        DRAM_table_start_address.insert(0, 0)

        HBM_table_start_address = [0 + vec_size * (collisions+len(hot_vec_loc[i])) for i in range(len(hot_vec_loc))]
        HBM_table_start_address.insert(0, 0)

        return DRAM_table_start_address, HBM_table_start_address

    def basic_physical_address_mapping(self, table_idx, vec_idx, is_r_vec=False, collisions=0):    
        ## r vec is located at the front of the table
        HBM_loc = False
        if vec_idx in self.hot_vec_loc[table_idx] or is_r_vec:
            logical_address = self.HBM_table_start_address[table_idx]
            table_start_vpn = int(logical_address // self.page_offset)        
            ppn = self.HBM_page_translation[table_start_vpn]
            HBM_loc = True
        else:
            logical_address = self.DRAM_table_start_address[table_idx]
            table_start_vpn = int(logical_address // self.page_offset)
            ppn = self.DRAM_page_translation[table_start_vpn]
        
        if HBM_loc:
            if not is_r_vec:
                physical_addr = int(ppn*self.page_offset + (collisions+vec_idx)*self.vec_size)
            else:
                physical_addr = int(ppn*self.page_offset + vec_idx*self.vec_size)
        else:
                physical_addr = int(ppn*self.page_offset + vec_idx*self.vec_size)

        return HBM_loc, physical_addr

    def bankgroup_based_logical_address_mapping(self, total_bankgroups, tables_per_bankgroup, hot_vec_loc, bankgroup_size, vector_size):
        '''
            performs logical address mapping regarding hot vector ranking and bank group size.
            fetch hot vectors from given tables_per_bankgroup, put them inside the same bank group.

            args:
                total_bankgroups : total bank groups
                tables_per_bankgroup : total tables per bank group
                hot_vec_loc : hot vector index list per table
                bankgroup_size : bank group size
                vector_size : vector size in Bytes

            returns :
                list of tuple per table. tuple consists of bankgroup_idx, start_q_vector_idx, end_q_vector_idx
                tuple : (bankgroup_idx, start_q_vector_idx, end_q_vector_idx)

            preprocess each bankgroup's ppn.
            check at what bankgroup is vector located
            perform logical -> physical translation by bankgroup's physical page address
        '''

        total_tables = len(hot_vec_loc)
        table_record = [[] for _ in range(total_tables)]
        bankgroup_record = [[] for _ in range(total_bankgroups)]
        vectors_left = [len(table) for table in hot_vec_loc]
        curr_table_idx = [i for i in range(tables_per_bankgroup)]

        vectors_per_table = bankgroup_size // tables_per_bankgroup // vector_size

        bankgroup_indicies = [i for i in range(total_bankgroups)]
        random.shuffle(bankgroup_indicies)

        for bankgroup in bankgroup_indicies:
            if len(curr_table_idx) == 0:
                break
            for table_idx in curr_table_idx:
                start_vec_idx = len(hot_vec_loc[table_idx]) - vectors_left[table_idx]
                vectors_left[table_idx] -= vectors_per_table
                if vectors_left[table_idx] < 0:
                    table_record[table_idx].append((bankgroup, start_vec_idx, start_vec_idx + vectors_per_table - abs(vectors_left[table_idx]) - 1))
                else:
                    table_record[table_idx].append((bankgroup, start_vec_idx, start_vec_idx + vectors_per_table - 1))

            for i in curr_table_idx:
                if not i in bankgroup_record[bankgroup]:
                    bankgroup_record[bankgroup].append(i)

            # remove and add new table index
            # when table has no vectors left, go to next available table

            remove_idx = []
            for i, table_idx in enumerate(curr_table_idx):
                if vectors_left[table_idx] <= 0:
                    remove_idx.append(i)
            last_index = curr_table_idx[-1]
            curr_table_idx = [table_idx for i, table_idx in enumerate(curr_table_idx) if i not in remove_idx]
            for idx in remove_idx:
                if last_index + 1 < total_tables:
                    curr_table_idx.append(last_index + 1)
                    last_index += 1

        return table_record, bankgroup_record

    def bankgroup_based_page_translation(self, bg_idx, ppn_idx):

        # addr mapping : {'rank' : 0, 'row' : 14, 'colhigh' : 5, 'channel' : 4, 'bankgroup' : 2, 'bank' : 4, 'collow' : 5, 'offset' : 3}

        # total bg # = rank_len * channel_len * bg_per_channel_len
        bankgroup_len = int(math.pow(2, AddressMapping.HBM_bit_width_4GB['bankgroup']))
        channel_len = int(math.pow(2, AddressMapping.HBM_bit_width_4GB['channel']))
        colhigh_len = int(math.pow(2, AddressMapping.HBM_bit_width_4GB['colhigh']))
        row_len = int(math.pow(2, AddressMapping.HBM_bit_width_4GB['row']))

        rank = bg_idx // (channel_len * bankgroup_len)
        channel = bg_idx % (channel_len * bankgroup_len)
        bankgroup = channel % bankgroup_len

        ppn = 0
        bit_stack = 12

        ppn = bankgroup * int(math.pow(2, bit_stack)) + ppn
        bit_stack += AddressMapping.HBM_bit_width_4GB['bankgroup']

        ppn = channel * int(math.pow(2, bit_stack)) + ppn
        bit_stack += AddressMapping.HBM_bit_width_4GB['channel']

        colhigh_idx = int(ppn_idx // math.pow(2, bit_stack-12)) % colhigh_len
        colhigh = self.bg_based_mapping_colhigh[colhigh_idx]
        ppn = colhigh * int(math.pow(2, bit_stack)) + ppn
        bit_stack += AddressMapping.HBM_bit_width_4GB['colhigh']

        row_idx = int(ppn_idx // math.pow(2, bit_stack-12)) % row_len
        row = self.bg_based_mapping_row[row_idx]
        ppn = row * int(math.pow(2, bit_stack)) + ppn
        bit_stack += AddressMapping.HBM_bit_width_4GB['row']

        ppn = rank * int(math.pow(2, bit_stack)) + ppn

        # print(rank, row, colhigh, channel, bankgroup)

        return ppn

    def bankgroup_based_physical_address_mapping(self, table_idx, vec_idx):
        if vec_idx in self.hot_vec_loc[table_idx]:
            HBM_loc = True
            bankgroup_idx = -1
            offset_in_bg = 0
            idx_in_hot_vec_loc = self.hot_vec_loc[table_idx].index(vec_idx)
            table_address_offsets = []
    
            # search for vec_idx's bankgroup
            for (bg_idx, start_vec, end_vec) in self.table_record[table_idx]:
                if start_vec <= idx_in_hot_vec_loc <= end_vec:
                    bankgroup_idx = bg_idx
                    offset_in_bg = idx_in_hot_vec_loc - start_vec

            # search for table index inside bg
            table_order = self.bg_record[bankgroup_idx].index(table_idx)
            table_address_in_bg = 0

            # search for table start address inside bg
            for i in range(table_order):
                table_id = self.bg_record[bankgroup_idx][i]
                for (bg_idx, start_vec, end_vec) in self.table_record[table_id]:
                    if bg_idx == bankgroup_idx:
                        table_address_in_bg += (end_vec - start_vec + 1)

            # get final address by performing (table address in bank group + offset from table address)
            # bg_size = [bank : col_low : offset] bits, address_in_one_bg <= bg_size
            address_in_one_bg = (table_address_in_bg + offset_in_bg) * self.vec_size 
            # if bg size > page_offset
            address_inside_page = address_in_one_bg % self.page_offset
            vpn_idx_in_one_bg = int(address_in_one_bg // self.page_offset)
            ppn = self.bankgroup_based_page_translation(bankgroup_idx, vpn_idx_in_one_bg)
            physical_addr = int(ppn*self.page_offset + address_inside_page)
            return HBM_loc, physical_addr

        else:
            HBM_loc = False
            logical_address = self.DRAM_table_start_address[table_idx]
            table_start_vpn = int(logical_address // self.page_offset)
            ppn = self.DRAM_page_translation[table_start_vpn]       
            physical_addr = int(ppn*self.page_offset + vec_idx*self.vec_size)

        return HBM_loc, physical_addr