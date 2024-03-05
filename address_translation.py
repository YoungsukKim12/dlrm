import math
import numpy as np
import random
import pickle
import sys
import os
from abc import *

class AddressTranslation():
    def __init__(self, embedding_profiles, hot_vector_total_access, collisions, translator_name):
        self.embedding_profiles = embedding_profiles
        self.hot_vector_total_access = hot_vector_total_access
        self.collisions = collisions
        self.translator_name = translator_name

    def mapper_name(self):
        return self.translator_name

    def profile_hot_vec_location(self):
        '''
            returns :  hot vector index list in each table
            
            hot vector index list is sorted for each table
        '''

        table_len = len(self.embedding_profiles)
        curr_idx_per_table = [0 for _ in range(table_len)]
        hot_vec_location = [[] for _ in range(table_len)]

        # hot_vector_savefile = './hot_vector_location_profile_w_collision_%d.pickle' %self.collisions

        # if os.path.exists(hot_vector_savefile):
        #     with open(hot_vector_savefile, 'rb') as loadfile:
        #         hot_vec_location = pickle.load(loadfile)

        # else:
        #     print("start calculating hot vector")
        total_access = 0
        hot_q_per_table = []
        hot_indices_per_table = []

        for i, prof_per_table in enumerate(self.embedding_profiles):
            total_access += np.sum(prof_per_table)
            q_access = np.sum(prof_per_table, axis=1)
            hot_q_indices = np.argsort(-q_access)
            hot_q_ranking = q_access[hot_q_indices]
            hot_indices_per_table.append(hot_q_indices)
            hot_q_per_table.append(hot_q_ranking)

        hot_vector_total_access = self.hot_vector_total_access

        if hot_vector_total_access == 1:
            for i in range(len(self.embedding_profiles)):
                hot_vec_location[i] = [j for j in range(len(self.embedding_profiles[i]))]
        else:
            while not hot_vector_total_access < 0:
                hot_vecs = [hot_q_ranking[curr_idx_per_table[table_id]] if not curr_idx_per_table[table_id] == -1 else -10 for table_id, hot_q_ranking in enumerate(hot_q_per_table)]
                top_hot_vec_table_id = np.argmax(hot_vecs)
                top_hot_vec_access_ratio = np.max(hot_vecs)
                top_hot_vec_idx_inside_table = np.where(hot_indices_per_table[top_hot_vec_table_id] == curr_idx_per_table[top_hot_vec_table_id])[0][0]
                hot_vec_location[top_hot_vec_table_id].append(top_hot_vec_idx_inside_table)
                curr_idx_per_table[top_hot_vec_table_id] += 1
                # all vectors in this table are used
                if len(hot_q_per_table[top_hot_vec_table_id]) == curr_idx_per_table[top_hot_vec_table_id]:
                    curr_idx_per_table[top_hot_vec_table_id] = -1

                hot_vector_total_access -= top_hot_vec_access_ratio

        total_vector = 0
        for table in self.embedding_profiles:
            total_vector += len(table)
        hot_vectors = 0
        for hot_vecs in hot_vec_location:
            hot_vectors += len(hot_vecs)

        print("total vector : ", total_vector)
        print("total hots : ", hot_vectors)
        

        return hot_vec_location

    def is_hot_vector(self, table_idx, vec_idx):
        return False

    # @abstractmethod
    # def physical_address_translation(self, table_idx, vec_idx, is_r_vec=False, collision_idx=0):    
    #     pass

class BasicAddressTranslation(AddressTranslation):

    def __init__(
        self, 
        embedding_profiles, 
        DIMM_size_gb=16, 
        vec_size=64,
        end_iter=20000,
        collisions=4,
        mapper_name="Basic"
    ):
        super().__init__(embedding_profiles, 1, collisions, mapper_name)
        self.vec_size = vec_size
        self.collisions = collisions
        self.page_offset = math.pow(2, 12)

        # DIMM size and ppns
        GB_size = math.pow(2, 30)
        DIMM_Size = DIMM_size_gb * GB_size
        DIMM_max_page_number = int(DIMM_Size // self.page_offset)

        # for basic address mapping (use random vpn -> ppn mapping)
        self.DIMM_page_translation = [i for i in range(DIMM_max_page_number)]
        random.shuffle(self.DIMM_page_translation)
        self.DIMM_table_start_address = self.basic_logical_address_translation(embedding_profiles, self.vec_size)

    def basic_logical_address_translation(self, embedding_profiles, vec_size):        
        DIMM_space_per_table = [vec_size * prof_per_table.shape[0] for prof_per_table in embedding_profiles]
        DIMM_table_start_address = []

        DIMM_accumulation = 0
        for space in DIMM_space_per_table:
            DIMM_table_start_address.append(DIMM_accumulation)
            DIMM_accumulation += space

        return DIMM_table_start_address

    def physical_address_translation(self, table_idx, vec_idx, is_r_vec, collision_idx=0):    
        table_start_logical_address = self.DIMM_table_start_address[table_idx]
        table_start_vpn = int(table_start_logical_address // self.page_offset)
        table_start_po = int(table_start_logical_address % self.page_offset)
        ppn = self.DIMM_page_translation[table_start_vpn]
        physical_addr = int(ppn * self.page_offset + table_start_po + vec_idx * self.vec_size)

        return False, physical_addr

class HeteroBasicAddressTranslation(AddressTranslation):

    def __init__(
        self, 
        embedding_profiles, 
        HBM_size_gb=4, 
        DIMM_size_gb=16, 
        hot_vector_total_access=0.9,
        vec_size = 64,
        end_iter=20000,
        collisions=4,
        r_load_balance=True,
        mapper_name = "HeteroBasic"
    ):
        super().__init__(embedding_profiles, hot_vector_total_access, collisions, mapper_name)
        self.is_QR = True
        self.vec_size = vec_size
        self.collisions = collisions
        self.page_offset = math.pow(2, 12)
        self.r_load_balance = r_load_balance
        self.hot_vec_loc = self.profile_hot_vec_location()
        self.mapper_name_ = mapper_name

        self.DIMM_table_start_address, self.HBM_table_start_address = self.basic_logical_address_translation(self.hot_vec_loc, self.embedding_profiles, self.vec_size, self.collisions)

        # DIMM size and ppns
        GB_size = math.pow(2, 30)
        HBM_Size = HBM_size_gb * GB_size
        DIMM_Size = DIMM_size_gb * GB_size
        HBM_max_page_number = int(HBM_Size // self.page_offset)
        DIMM_max_page_number = int(DIMM_Size // self.page_offset)
        # for basic address mapping (use random vpn -> ppn mapping)
        if self.r_load_balance:
            # must be carefully calculated regarding address mapping
            total_r_size = self.vec_size * self.collisions * len(embedding_profiles)
            total_pages_for_r = int(total_r_size // self.page_offset)
            self.HBM_page_translation = [i for i in range(total_pages_for_r, HBM_max_page_number)]
        else:
            self.HBM_page_translation = [i for i in range(HBM_max_page_number)]
        self.DIMM_page_translation = [i for i in range(DIMM_max_page_number)]
        random.shuffle(self.HBM_page_translation)
        random.shuffle(self.DIMM_page_translation)

    def basic_logical_address_translation(self, hot_vec_loc, embedding_profiles, vec_size, collisions):        
        DIMM_space_per_table = [0 + vec_size * (prof_per_table.shape[0]-len(hot_vec_loc[i])) for i, prof_per_table in enumerate(embedding_profiles)]
        HBM_space_per_table = [0 + vec_size * (collisions+len(hot_vec_loc[i])) for i in range(len(hot_vec_loc))]
        if self.mapper_name_ == "SPACE":
            HBM_space_per_table = [0 + vec_size * (collisions+len(hot_vec_loc[i])*collisions) for i in range(len(hot_vec_loc))]

        if self.r_load_balance:
            HBM_space_per_table = [0 + vec_size * (len(hot_vec_loc[i])) for i in range(len(hot_vec_loc))]

        DIMM_table_start_address = []
        HBM_table_start_address = []
        DIMM_accumulation = 0
        for i in range(len(DIMM_space_per_table)):
            DIMM_table_start_address.append(DIMM_accumulation)
            DIMM_accumulation += DIMM_space_per_table[i]

        HBM_accumulation = 0
        for i in range(len(HBM_space_per_table)):
            HBM_table_start_address.append(HBM_accumulation)
            HBM_accumulation += HBM_space_per_table[i]
        print("total vectors stored in HBM in GB: ", HBM_accumulation/1024/1024/1024)

        return DIMM_table_start_address, HBM_table_start_address

    def physical_address_translation(self, table_idx, vec_idx, is_r_vec=False, collision_idx=0):    
        ## r vec is located at the front of the table
        HBM_loc = False

        # generate ppn
        if vec_idx in self.hot_vec_loc[table_idx] or is_r_vec:
            table_start_logical_address = self.HBM_table_start_address[table_idx]
            table_start_vpn = int(table_start_logical_address // self.page_offset)        
            table_start_po = int(table_start_logical_address % self.page_offset)
            ppn = self.HBM_page_translation[table_start_vpn]
            # if self.r_load_balance and is is_r_vec:
            #     ppn = (table_idx * collisions * self.vec_size + vec_idx * self.vec_size) // self.page_offset
            HBM_loc = True
        else:
            table_start_logical_address = self.DIMM_table_start_address[table_idx]
            table_start_vpn = int(table_start_logical_address // self.page_offset)
            table_start_po = int(table_start_logical_address % self.page_offset)
            ppn = self.DIMM_page_translation[table_start_vpn]

        # generate physical address        
        if HBM_loc:
            if self.collisions > 0:
                if not is_r_vec:
                    physical_addr = int(ppn*self.page_offset + table_start_po + (self.collisions+vec_idx)*self.vec_size)
                    if self.mapper_name_ == "SPACE":
                        physical_addr = int(ppn*self.page_offset + table_start_po + (self.collisions+vec_idx*self.collisions+collision_idx)*self.vec_size)
                else:
                    physical_addr = int(ppn*self.page_offset + table_start_po + vec_idx*self.vec_size)
                    if self.r_load_balance:
                        physical_addr = int(table_idx * self.collisions * self.vec_size + vec_idx * self.vec_size)
            else:
                physical_addr = int(ppn*self.page_offset + table_start_po + vec_idx*self.vec_size)
        else:
                physical_addr = int(ppn*self.page_offset + table_start_po + vec_idx*self.vec_size)

        return HBM_loc, physical_addr

class RecNMPAddressTranslation(AddressTranslation):
    def __init__(
        self, 
        embedding_profiles, 
        DIMM_size_gb=8, 
        hot_vector_total_access=0.05,
        vec_size=64,
        collisions=4,
        mapper_name="RecNMP"
    ):  
        super().__init__(embedding_profiles, hot_vector_total_access, collisions, mapper_name)
        self.vec_size = vec_size
        self.collisions = collisions
        self.page_offset = math.pow(2, 12)
        self.DIMM_table_start_address = self.logical_address_translation(self.embedding_profiles, self.vec_size)
        self.hot_vec_loc = self.profile_hot_vec_location()

        # DIMM size and ppns
        GB_size = math.pow(2, 30)
        DIMM_Size = DIMM_size_gb * GB_size
        DIMM_max_page_number = int(DIMM_Size // self.page_offset)

        # For address mapping (use random vpn -> ppn mapping)
        self.DIMM_page_translation = [i for i in range(DIMM_max_page_number)]
        random.shuffle(self.DIMM_page_translation)

    def logical_address_translation(self, embedding_profiles, vec_size):
        DIMM_space_per_table = [vec_size * (self.collisions + prof_per_table.shape[0]) for prof_per_table in embedding_profiles]
        DIMM_table_start_address = []

        DIMM_accumulation = 0
        for space in DIMM_space_per_table:
            DIMM_table_start_address.append(DIMM_accumulation)
            DIMM_accumulation += space

        print("total vectors stored in RecNMP in GB: ", DIMM_accumulation/1024/1024/1024)


        return DIMM_table_start_address

    def is_hot_vector(self, table_idx, vec_idx):
        if vec_idx in self.hot_vec_loc[table_idx]:
            return True
        else:
            return False

    def physical_address_translation(self, table_idx, vec_idx, is_r_vec=False, collision_idx=0):
        table_start_logical_address = self.DIMM_table_start_address[table_idx]
        table_start_vpn = int(table_start_logical_address // self.page_offset)
        table_start_po = int(table_start_logical_address % self.page_offset)
        ppn = self.DIMM_page_translation[table_start_vpn]

        if is_r_vec:
            physical_addr = int(ppn * self.page_offset + table_start_po + vec_idx * self.vec_size)
        else:
            physical_addr = int(ppn * self.page_offset + table_start_po + (self.collisions + vec_idx) * self.vec_size)

        return False, physical_addr

class TRiMAddressTranslation(AddressTranslation):
    def __init__(
        self, 
        embedding_profiles, 
        DIMM_size_gb=16,
        hot_vector_total_access=0.05,
        vec_size=64,
        end_iter=20000,
        collisions=4,
        bank_group_bits_naive=27,
        total_bankgroups=4*4,
        mapper_name="TRiM"
    ):
        self.total_bankgroups = total_bankgroups
        self.bank_group_bits = bank_group_bits_naive
        self.embedding_profiles = embedding_profiles
        self.vec_size = vec_size
        self.collisions = collisions
        self.adjusted_hot_vector_total_access = self.adjust_hot_vector_total_access(embedding_profiles, hot_vector_total_access, collisions)
        super().__init__(embedding_profiles, self.adjusted_hot_vector_total_access, collisions, mapper_name)
        self.total_r_vectors = self.calculate_r_vector_count()
        self.hot_vec_loc = self.profile_hot_vec_location()
        self.logical_to_physical, self.r_vector_mapping, self.q_vector_mapping = self.logical2physicalmapping()
        # print(self.r_vector_mapping)

    def adjust_hot_vector_total_access(self, embedding_profiles, hot_vector_total_access, collisions):
        total_vectors = sum(len(table) for table in embedding_profiles)
        total_r_vectors = len(embedding_profiles) * collisions
        adjusted_hot_vector_total_access = max(hot_vector_total_access - (total_r_vectors / total_vectors), 0)
        return adjusted_hot_vector_total_access

    def calculate_r_vector_count(self):
        total_tables = len(self.embedding_profiles)
        return total_tables * self.collisions

    def is_hot_vector(self, table_idx, vec_idx):
        if vec_idx in self.hot_vec_loc[table_idx]:
            return True
        else:
            return False

    def logical2physicalmapping(self):
        max_k_bits = 2 ** self.bank_group_bits
        bank_group_size = max_k_bits  # Not the actual physical size, used for address mapping

        # Reserve space for r vectors and hot q vectors
        total_hot_q_vectors = sum(len(hot_vecs) for hot_vecs in self.hot_vec_loc)
        reserved_r_space = self.total_r_vectors * self.vec_size
        reserved_q_space = total_hot_q_vectors * self.vec_size

        r_vector_mapping = {}
        q_vector_mapping = {}
        logical_to_physical = {}

        # Fill reserved space for r vectors
        current_physical_address = 0
        for i in range(self.total_r_vectors):
            r_vector_mapping[i] = current_physical_address
            current_physical_address += self.vec_size

        # Fill reserved space for q vectors
        for i in range(total_hot_q_vectors):
            q_vector_mapping[i] = current_physical_address
            current_physical_address += self.vec_size

        print("total Bytes reserved for hot entries :", current_physical_address)

        # Mapping for other vectors
        for table_idx, table in enumerate(self.embedding_profiles):
            for item_idx in range(len(table)):
                if item_idx in self.hot_vec_loc[table_idx]:
                    continue

                bank_group_idx = current_physical_address // bank_group_size
                if bank_group_idx < self.total_bankgroups:
                    start_address_for_bank_group = bank_group_idx * bank_group_size + reserved_r_space + reserved_q_space
                else:
                    start_address_for_bank_group = bank_group_idx * bank_group_size

                if current_physical_address < start_address_for_bank_group:
                    current_physical_address = start_address_for_bank_group


                logical_to_physical[(table_idx, item_idx)] = current_physical_address
                current_physical_address += self.vec_size

        print("total vectors stored in TRiM in GB: ", current_physical_address/1024/1024/1024)

        return logical_to_physical, r_vector_mapping, q_vector_mapping

    def physical_address_translation(self, table_idx, vec_idx, is_r_vec=False, collision_idx=0):
        vec_idx = int(vec_idx)
        if is_r_vec:
            return False, self.r_vector_mapping.get(self.collisions*table_idx + vec_idx, None)
        elif vec_idx in self.hot_vec_loc[table_idx]:
            q_vector_global_idx = sum(len(hot_vecs) for hot_vecs in self.hot_vec_loc[:table_idx]) + self.hot_vec_loc[table_idx].index(vec_idx)
            return False, self.q_vector_mapping.get(q_vector_global_idx, None)
        else:
            return False, self.logical_to_physical.get((table_idx, vec_idx), None)