import math
import numpy as np
import random
import pickle
import sys
import os
from abc import *

class HBMInfo():
    def __init__(self, GB_Size):
        self.HBM_Size = GB_Size
        self.bit_width = {'rank' : 0, 'row' : 14, 'colhigh' : 5, 'channel' : 0, 'bankgroup' : 2, 'bank' : 4, 'collow' : 5, 'offset' : 3}
        self.generate_bit_width()

    def generate_bit_width(self):
        size_in_byte = self.HBM_Size * math.pow(2, 30)
        DRAM_dies = self.HBM_Size
        channel = DRAM_dies * 4
        rank = DRAM_dies // 4

        self.bit_width['channel'] = math.log2(channel)
        self.bit_width['rank'] = math.log2(rank)

class AddressMapping():
    def __init__(self, profiles, hot_access_ratio, collisions):
        self.profiles = profiles
        self.hot_access_ratio = hot_access_ratio
        self.collisions = collisions

    def profile_hot_vec_location(self):
        '''
            returns :  hot vector index list of each table
            
            hot vector index list is sorted for each table
        '''

        table_len = len(self.profiles)
        curr_idx_per_table = [0 for _ in range(table_len)]
        hot_vec_location = [[] for _ in range(table_len)]

        hot_vector_savefile = 'savedata/hot_vector_location_profile_w_collision_%d.pickle' %self.collisions

        if os.path.exists(hot_vector_savefile):
            with open(hot_vector_savefile, 'rb') as loadfile:
                hot_vec_location = pickle.load(loadfile)

        else:
            total_access = 0
            hot_q_per_table = []
            hot_indices_per_table = []

            for i, prof_per_table in enumerate(self.profiles):
                total_access += np.sum(prof_per_table)
                q_access = np.sum(prof_per_table, axis=1)

                hot_q_indices = np.argsort(-q_access)
                hot_q_ranking = q_access[hot_q_indices]
                hot_indices_per_table.append(hot_q_indices)
                hot_q_per_table.append(hot_q_ranking)

            hot_access_ratio = self.hot_access_ratio

            while not hot_access_ratio < 0:
                hot_vecs = [int(hot_q_ranking[curr_idx_per_table[table_id]]) if not curr_idx_per_table[table_id] == -1 else -10 for table_id, hot_q_ranking in enumerate(hot_q_per_table)]
                top_hot_vec_table_id = np.argmax(hot_vecs)
                top_hot_vec_access_ratio = np.max(hot_vecs)
                top_hot_vec_idx_inside_table = np.where(hot_indices_per_table[top_hot_vec_table_id] == curr_idx_per_table[top_hot_vec_table_id])[0][0]
                hot_vec_location[top_hot_vec_table_id].append(top_hot_vec_idx_inside_table)
                curr_idx_per_table[top_hot_vec_table_id] += 1

                # all vectors in this table are used
                if len(hot_q_per_table[top_hot_vec_table_id]) == curr_idx_per_table[top_hot_vec_table_id]:
                    curr_idx_per_table[top_hot_vec_table_id] = -1

                hot_access_ratio -= top_hot_vec_access_ratio / total_access

            with open(hot_vector_savefile, 'wb') as savefile:
                pickle.dump(hot_vec_location, savefile)

        return hot_vec_location

    @abstractmethod
    def physical_address_mapping(self, table_idx, vec_idx, is_r_vec=False):
        pass

class BasicAddressMapping(AddressMapping):

    def __init__(
        self, 
        profiles, 
        HBM_size_gb=4, 
        DIMM_size_gb=8, 
        hot_access_ratio=0.7,
        vec_size = 64,
        end_iter=20000,
        collisions=4
    ):
        super().__init__(profiles, hot_access_ratio, collisions)
        self.is_QR = True
        self.vec_size = vec_size
        self.collisions = collisions
        self.page_offset = math.pow(2, 12)
        self.hot_vec_loc = self.profile_hot_vec_location()
        self.DRAM_table_start_address, self.HBM_table_start_address = self.basic_logical_address_mapping(self.hot_vec_loc, self.profiles, self.vec_size, self.collisions)

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

    def basic_logical_address_mapping(self, hot_vec_loc, profiles, vec_size, collisions):        
        DRAM_space_per_table = [0 + vec_size * (prof_per_table.shape[0]-len(hot_vec_loc[i])) for i, prof_per_table in enumerate(profiles)]
        HBM_space_per_table = [0 + vec_size * (collisions+len(hot_vec_loc[i])) for i in range(len(hot_vec_loc))]
        DRAM_table_start_address = []
        HBM_table_start_address = []

        accumulation = 0
        for i in range(len(DRAM_space_per_table)):
            DRAM_table_start_address.append(accumulation)
            accumulation += DRAM_space_per_table[i]

        accumulation = 0
        for i in range(len(HBM_space_per_table)):
            HBM_table_start_address.append(accumulation)
            accumulation += HBM_space_per_table[i]

        ################################## Test ##################################

        # for i, prof_per_table in enumerate(profiles):
        #     print("profiles in table %d : %d" %(i, prof_per_table.shape[0]))

        # for i, hot_vecs in enumerate(hot_vec_loc):
        #     print("hot vecs in table %d : %d" %(i, len(hot_vec_loc[i])))

        # for i in range(len(profiles)):
        #     print("vectors of table %d : %d" %(i, profiles[i].shape[0] - len(hot_vec_loc[i])))

        # for i in range(len(DRAM_table_start_address)):
        #     if i > 0:
        #         print("vectors between tables : ", int((DRAM_table_start_address[i] - DRAM_table_start_address[i-1])/vec_size))

        return DRAM_table_start_address, HBM_table_start_address

    def physical_address_mapping(self, table_idx, vec_idx, is_r_vec=False):    
        ## r vec is located at the front of the table
        HBM_loc = False

        # generate ppn
        if vec_idx in self.hot_vec_loc[table_idx] or is_r_vec:
            logical_address = self.HBM_table_start_address[table_idx]
            table_start_vpn = int(logical_address // self.page_offset)        
            ppn = self.HBM_page_translation[table_start_vpn]
            HBM_loc = True
        else:
            logical_address = self.DRAM_table_start_address[table_idx]
            table_start_vpn = int(logical_address // self.page_offset)
            ppn = self.DRAM_page_translation[table_start_vpn]


        # generate physical address        
        if HBM_loc:
            if self.collisions > 0:
                if not is_r_vec:
                    physical_addr = int(ppn*self.page_offset + (self.collisions+vec_idx)*self.vec_size)
                else:
                    physical_addr = int(ppn*self.page_offset + vec_idx*self.vec_size)
            else:
                physical_addr = int(ppn*self.page_offset + vec_idx*self.vec_size)
        else:
                physical_addr = int(ppn*self.page_offset + vec_idx*self.vec_size)

        ################################## Test ##################################

        # if not HBM_loc:
        #     extracted_ppn = (physical_addr - vec_idx*self.vec_size) / self.page_offset
        #     extracted_table_start_vpn = self.DRAM_page_translation.index(extracted_ppn)
        #     print("extracted ppn : %d" %extracted_ppn)
        #     print("extracted table start vpn : %d" %(extracted_table_start_vpn))

        #     if(extracted_ppn == ppn and extracted_table_start_vpn == table_start_vpn):
        #         print("test pass : vec in DRAM")
        #     else:
        #         print("test fail on DRAM")
        #         sys.exit()
        # else:
        #     if not is_r_vec:
        #         extracted_ppn = (physical_addr - (self.collisions+vec_idx)*self.vec_size) / self.page_offset
        #         extracted_table_start_vpn = self.HBM_page_translation.index(extracted_ppn)
        #         print("extracted ppn : %d" %extracted_ppn)
        #         print("extracted table start vpn : %d" %(extracted_table_start_vpn))

        #         if(extracted_ppn == ppn and extracted_table_start_vpn == table_start_vpn):
        #             print("test pass : q vec in HBM")            
        #         else:
        #             print("test fail on q vec in HBM")
        #             sys.exit()
        #     else:
        #         extracted_ppn = (physical_addr - vec_idx*self.vec_size) / self.page_offset
        #         extracted_table_start_vpn = self.HBM_page_translation.index(extracted_ppn)
        #         print("extracted ppn : %d" %extracted_ppn)
        #         print("extracted table start vpn : %d" %(extracted_table_start_vpn))

        #         if(extracted_ppn == ppn and extracted_table_start_vpn == table_start_vpn):
        #             print("test pass : r vec in HBM")            
        #         else:
        #             print("test fail on r vec in HBM")
        #             sys.exit()

        return HBM_loc, physical_addr

class BGAddressMapping(AddressMapping):
    def __init__(
        self, 
        profiles, 
        HBM_size_gb=4, 
        DIMM_size_gb=8, 
        hot_access_ratio=0.7,
        vec_size = 64,
        end_iter=20000,
        collisions = 4,
        tables_per_bankgroup=3
    ):

        super().__init__(profiles, hot_access_ratio, collisions)
        # basic parameters
        self.profiles = profiles
        self.hot_access_ratio = hot_access_ratio
        self.vec_size = vec_size
        self.page_offset = math.pow(2, 12)
        self.tables_per_bankgroup = 3
        self.collisions = collisions

        # DRAM size and ppns
        GB_size = math.pow(2, 30)
        HBM_Size = HBM_size_gb * GB_size
        DRAM_Size = DIMM_size_gb * GB_size
        HBM_max_page_number = int(HBM_Size // self.page_offset)
        DRAM_max_page_number = int(DRAM_Size // self.page_offset)
        HBM_ppn_bits = int(math.log2(HBM_max_page_number))
        self.HBM_bits = HBMInfo(HBM_size_gb)

        # for bg based address mapping
        self.total_bankgroups = int(math.pow(2, (self.HBM_bits.bit_width['rank'] + self.HBM_bits.bit_width['channel'] + self.HBM_bits.bit_width['bankgroup'])))
        self.bankgroup_size = int(HBM_Size // self.total_bankgroups)
        print("total bank groups : ", self.total_bankgroups)
        print("bankgroup size in bytes : ", self.bankgroup_size)

        # preprocess
        self.hot_vec_loc = self.profile_hot_vec_location()
        self.table_record, self.bg_record = self.bankgroup_based_logical_address_mapping(self.total_bankgroups, self.tables_per_bankgroup, self.hot_vec_loc, self.bankgroup_size, self.vec_size)
        self.DRAM_table_start_address = self.basic_logical_address_mapping(self.hot_vec_loc, self.profiles, self.vec_size)

        self.DRAM_page_translation = [i for i in range(DRAM_max_page_number)]
        random.shuffle(self.DRAM_page_translation)


        self.bg_based_mapping_row = [i for i in range(int(math.pow(2, self.HBM_bits.bit_width['row'])))]
        self.bg_based_mapping_colhigh = [i for i in range(int(math.pow(2, self.HBM_bits.bit_width['colhigh'])))]
        random.shuffle(self.bg_based_mapping_row)
        random.shuffle(self.bg_based_mapping_colhigh)

        print('table_record', self.table_record)
        print('bankgroup_record', self.bg_record)

        # for i in range(len(self.hot_vec_loc)):
        #     print("table # %d hot vecs : %d" %(i, len(self.hot_vec_loc[i])))

    def basic_logical_address_mapping(self, hot_vec_loc, profiles, vec_size):        
        DRAM_space_per_table = [0 + vec_size * (prof_per_table.shape[0]-len(hot_vec_loc[i])) for i, prof_per_table in enumerate(profiles)]
        DRAM_table_start_address = []

        accumulation = 0
        for i in range(len(DRAM_space_per_table)):
            DRAM_table_start_address.append(accumulation)
            accumulation += DRAM_space_per_table[i]


        return DRAM_table_start_address


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
                list of tuple per table.
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

        # offload vectors of table to bankgroup regarding bankgroup size
        # table indexes are incremented sequentially
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

    def bankgroup_based_page_translation(self, bg_idx, vpn_idx):

        # addr mapping : {'rank' : 0, 'row' : 14, 'colhigh' : 5, 'channel' : 4, 'bankgroup' : 2, 'bank' : 4, 'collow' : 5, 'offset' : 3}

        # total bg # = rank_len * channel_len * bg_per_channel_len
        bankgroup_len = int(math.pow(2, self.HBM_bits.bit_width['bankgroup']))
        channel_len = int(math.pow(2, self.HBM_bits.bit_width['channel']))
        colhigh_len = int(math.pow(2, self.HBM_bits.bit_width['colhigh']))
        row_len = int(math.pow(2, self.HBM_bits.bit_width['row']))

        rank = bg_idx // (channel_len * bankgroup_len)
        channel = bg_idx % (channel_len * bankgroup_len)
        bankgroup = channel % bankgroup_len

        ppn = 0
        bit_stack = 12

        ppn = bankgroup * int(math.pow(2, bit_stack)) + ppn
        bit_stack += self.HBM_bits.bit_width['bankgroup']

        ppn = channel * int(math.pow(2, bit_stack)) + ppn
        bit_stack += self.HBM_bits.bit_width['channel']

        colhigh_idx = int(vpn_idx // math.pow(2, bit_stack-12)) % colhigh_len
        colhigh = self.bg_based_mapping_colhigh[colhigh_idx]
        ppn = colhigh * int(math.pow(2, bit_stack)) + ppn
        bit_stack += self.HBM_bits.bit_width['colhigh']

        row_idx = int(vpn_idx // math.pow(2, bit_stack-12)) % row_len
        row = self.bg_based_mapping_row[row_idx]
        ppn = row * int(math.pow(2, bit_stack)) + ppn
        bit_stack += self.HBM_bits.bit_width['row']

        ppn = rank * int(math.pow(2, bit_stack)) + ppn

        ####################   Test   #########################
        # col_high test keeps failing by minimal difference in value but other things are fine.        
        # check row, colhigh, channel, bankgroup value by dividing ppn with bit stacks
        # tmp_bit_stack = bit_stack
        # tmp_ppn = ppn
        # if rank == tmp_ppn // int(math.pow(2, tmp_bit_stack)):
        #     print("rank test pass")
        #     tmp_ppn -= rank * int(math.pow(2, tmp_bit_stack))
        #     tmp_bit_stack -= self.HBM_bits.bit_width['row']
        # else:
        #     print("rank : %d, calculated rank : %d" %(rank, tmp_ppn // int(math.pow(2, tmp_bit_stack))))

        # if row == tmp_ppn // int(math.pow(2, tmp_bit_stack)):
        #     print("row test pass")
        #     tmp_ppn -= row * int(math.pow(2, tmp_bit_stack))
        #     tmp_bit_stack -= self.HBM_bits.bit_width['colhigh']
        # else:
        #     print("row : %d, calculated row : %d" %(row, tmp_ppn // int(math.pow(2, tmp_bit_stack))))

        # if colhigh == tmp_ppn // int(math.pow(2, tmp_bit_stack)):
        #     print("col test pass")
        #     tmp_ppn -= colhigh * int(math.pow(2, tmp_bit_stack))
        #     tmp_bit_stack -= self.HBM_bits.bit_width['channel']
        # else:
        #     print("col test fail - colhigh : %d, calculated colhigh : %d" %(colhigh, tmp_ppn // int(math.pow(2, tmp_bit_stack))))
        #     tmp_ppn -= colhigh * int(math.pow(2, tmp_bit_stack))
        #     tmp_bit_stack -= self.HBM_bits.bit_width['channel']

        # if channel == tmp_ppn // int(math.pow(2, tmp_bit_stack)):
        #     print("channel test pass")
        #     tmp_ppn -= channel * int(math.pow(2, tmp_bit_stack))
        #     tmp_bit_stack -= self.HBM_bits.bit_width['bankgroup']
        # else:
        #     print("channel test fail - : %d, calculated channel : %d" %(channel, tmp_ppn // int(math.pow(2, tmp_bit_stack))))
        #     tmp_ppn -= channel * int(math.pow(2, tmp_bit_stack))
        #     tmp_bit_stack -= self.HBM_bits.bit_width['bankgroup']

        # if bankgroup == tmp_ppn // int(math.pow(2, tmp_bit_stack)):
        #     print("bankgroup test pass")

        return ppn

    def physical_address_mapping(self, table_idx, vec_idx, is_r_vec=False):

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

            ####################   Test   #########################
            # check table start address of each table
            # print("table # %d address : %d bg, %d" %(table_idx, table_address_in_bg, bankgroup_idx))

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


# same as basic address mapping if not using partial sum
class SpaceAddressMapping():
    def __init__(
        self, 
        profiles, 
        HBM_size_gb=4, 
        DIMM_size_gb=8, 
        hot_access_ratio=0.7,
        vec_size = 64,
        end_iter=20000, 
    ):

        super().__init__()
        self.HBM_size_in_bytes = HBM_size_gb * math.pow(2, 30)
        self.hot_vec_loc = self.profile_hot_vec_location(self.profiles, self.hot_access_ratio)
        self.DRAM_table_start_address, self.HBM_table_start_address = self.basic_logical_address_mapping(self.hot_vec_loc, self.profiles, self.vec_size, self.collisions)
        self.partial_sums = self.partial_sum_address_mapping(self.HBM_table_start_address, HBM_size_in_bytes, vec_size)

    def combination(self, n:int, k:int):
        n_fac = math.factorial(n)
        k_fac = math.factorial(k)
        n_minus_k_fac = math.factorial(n-k)
        total_combination = n_fac / (k_fac*n_minus_k_fac)

        return total_combination

    def basic_logical_address_mapping(self, hot_vec_loc, profiles, vec_size, collisions):
        
        for hot_vecs in hot_vec_loc:
            print("hot vecs in table %d : %d" %(i, hot_vec_loc[i]))

        DRAM_table_start_address = [0 + vec_size * (prof_per_table.shape[0]-len(hot_vec_loc[i])) for i, prof_per_table in enumerate(profiles)]
        DRAM_table_start_address.insert(0, 0)

        for i in range(len(DRAM_table_start_address)):
            print("table start address :", DRAM_table_start_address[i])
            if i > 1:
                print("vectors between tables : ", (DRAM_table_start_address[i] - DRAM_table_start_address[i-1])/vec_size)


        HBM_table_start_address = [0 + vec_size * (collisions+len(hot_vec_loc[i])) for i in range(len(hot_vec_loc))]
        HBM_table_start_address.insert(0, 0) 

        return DRAM_table_start_address, HBM_table_start_address

    def basic_physical_address_mapping(self, table_idx, vec_idx, is_r_vec=False, collisions=0):
        ## r vec is located at the front of the table

        # generate ppn
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

        # generate physical address        
        if HBM_loc:
            if not is_r_vec:
                physical_addr = int(ppn*self.page_offset + (collisions+vec_idx)*self.vec_size)
            else:
                physical_addr = int(ppn*self.page_offset + vec_idx*self.vec_size)
        else:
                physical_addr = int(ppn*self.page_offset + vec_idx*self.vec_size)

        return HBM_loc, physical_addr    


'''
    def partial_sum_address_mapping(self, HBM_table_start_address, HBM_size_in_bytes, vec_size):

        used_HBM_memory = HBM_table_start_address[-1]
        available_HBM_memory = HBM_size_in_bytes - used_HBM_memory
        print('space for psums: ', available_HBM_memory)

        psum_range = 0
        while(True):
            total_combination = self.combination(psum_range, 2)
            if available_HBM_memory <= total_combination * vec_size:
                psum_range -= 1
                break
            else:
                psum_range += 1

        psums = []
        for i in range(psum_range):
            for j in range(psum_range):
                if j > i:
                    psums.append((i, j))

        random.shuffle(psums)

        return psums
    def check_and_convert_to_psum(self, total_embeddings):
            search for psum elements in embedding operation and convert them to psums
            returns non-psum elements in embedding operation
            only convert q vectors to psums

#        for table_idx, vec_idx in total_embeddings:
#            if vec_idx in self.hot_vec_loc[table_idx] and :
                
        pass
'''