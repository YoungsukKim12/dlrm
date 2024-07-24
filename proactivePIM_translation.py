import math
import numpy as np
import random
import pickle
import sys
import os
from abc import *

class WeightSharingTranslator():
    def __init__(
        self,
        embedding_profiles,
        collision,
        rank,
        vec_size=[256, 512, 1024, 2048]
    ):

    self.embedding_profiles = embedding_profiles
    self.collision = collision
    self.rank = rank

    self.Q_entries_per_table, R_entries_per_table = self.preprocess_QR()
    self.core_info = self.preprocess_TT_Rec()

    def preprocess_QR(self):
        Q_entries_per_table = [self.embedding_profiles[i]//self.collision for i in range(len(self.embedding_profiles))]
        R_entries_per_table = [self.collision for i in range(len(self.embedding_profiles))]

        return Q_entries_per_table, R_entries_per_table

    def preprocess_TT_Rec(self):
        core_entries_per_table = []
        vec_dim_per_table = []
        for i in range(len(self.embedding_profiles)):
            entry = self.find_number_and_combination_entries(self.embedding_profiles[i])
            core_entries_per_table.append(entry)

        for j in range(len(self.vec_size)):
            vec_dims = self.find_number_and_combination_for_vec(self.vec_size[j])      
            vec_dim_per_table.append(vec_dims)

        core_info = {}
        for k in range(len(self.vec_size)):
            core_info[self.vec_size[k]] = [vec_dim_per_table[k], core_entries_per_table]

        return core_info

    def get_QR_size(self, table_idx, vec_size, is_Q):
        if is_Q:
            return Q_entries_per_table[table_idx] * vec_size
        else:
            return R_entries_per_table[table_idx] * vec_size

    def get_TT_Rec_size(self, table_idx, vec_size, is_first, is_second, is_third):
        core_dims, cores_entries_per_table = self.core_info[vec_size]

        core_dim = 0
        cores_entries = core_entries_per_table[table_idx]
        if is_first:
            core_dim = core_dims[0] * cores_entries * self.rank * 4
        elif is_second:
            core_dim = core_dims[1] * cores_entries * self.rank * self.rank * 4 * 4
        elif is_third:
            core_dim = core_dims[2] * cores_entries * self.rank * 4

        return core_dim

    def get_QR_entry(self, vec_size, table_idx, vec_idx):
        return Q_entries_per_table[table_idx][vec_idx], vec_idx // self.collision

    def get_TT_Rec_entry(self, vec_size, table_idx, vec_idx):
        core_dims, core_entries_per_table = self.core_info[vec_size]
        cores_entries = core_entries_per_table[table_idx]
        access = []
        for i in range(vec_size):
            a = vec_idx % cores_entries * core_dims[0] + i % core_dims[0]
            b = ((vec_idx % cores_entries) // cores_entries) * ((vec_size % core_dims[2]) // core_dims[1]) + ((i % core_dims[2]) // core_dims[1])
            c = vec_idx // cores_entries * (vec_size // core_dims[2]) + i // core_dims[2]
            access.append((a,b,c))

        return access

    def find_number_and_combination_entries(self, N):
        x = 1
        while (x + 1) ** 3 <= N:
            x += 1

        largest_x_cubed = x ** 3
        print(f"The largest number that can be expressed as x^3 <= {N} is: {largest_x_cubed} (x={x})")

        y = x
        while (y ** 3) < N:
            y += 1
        print(f"The new combination by incrementing the number until its cube is >= {N} is: {y} with cube: {new_combination_value}")

        new_combination_value = y ** 3
        
        return y

    def find_number_and_combination_for_vec(self, N):
        x = 2
        while (x + 2) ** 3 <= N:
            x += 2

        largest_x_cubed = x ** 3
        print(f"The largest number that can be expressed as (2k)^3 <= {N} is: {largest_x_cubed} (x={x})")

        # Step 2: Increment x by 2 and find new combination
        y1, y2, y3 = x, x, x
        i = 0
        while y1 * y2 * y3 < N:
            if i==0:
                y1 += 2
            elif i==1:
                y2 += 2
            else:
                y3 += 2

            i = (i+1)%3

        new_combination_product = y1 * y2 * y3
        
        print(f"The new combination by incrementing each number by 2 until their product is >= {N} is: ({y1}, {y2}, {y3}) with product: {new_combination_product}")

        return (y1, y2, y3)

class ProactivePIMTranslation():
    def __init__(
        self, 
        embedding_profiles, 
        HBM_size_gb=4, 
        HBM_BW=256,
        vec_size=64,
        is_QR=True,
        collisions=8,
        is_TT_Rec=False,
        rank=16,
        mapper_name="ProactivePIM"
        # DIMM_size_gb=16, 
        # DIMM_BW=25.6,
    ):
        self.is_QR = is_QR
        self.is_TT_Rec = is_TT_Rec
        self.vec_size = vec_size
        self.collisions = collisions
        self.rank = rank
        
        self.nodes = HBM_size_gb // 0.5
        self.page_offset = math.pow(2, 12)
        self.hot_vec_loc = self.profile_hot_vec_location()
        self.mapper_name_ = mapper_name

        # DIMM size and ppns
        self.GB_size = math.pow(2, 30)
        self.HBM_Size = HBM_size_gb * self.GB_size
        self.DIMM_Size = DIMM_size_gb * self.GB_size
        self.HBM_max_page_number = int(self.HBM_Size // self.page_offset)
        self.DIMM_max_page_number = int(self.DIMM_Size // self.page_offset)

        # preprocess for logical2physical translation
        self.ws_translator = WeightSharingTranslator(embedding_profiles, collisions, rank)
        self.preprocess()

    def preprocess(self):
        # Reserve space for subtable duplication
        self.r_size = 0
        self.first_size = 0
        self.third_size = 0
        prefetch_info_filename = f"prefetch_info_{vec_size}"

        if self.is_QR:
            self.r_size_per_table = self.vec_size * self.collisions
            self.reserved_page = int(self.r_size_per_table*len(embedding_profiles) // self.page_offset) * self.nodes
            prefetch_info_filename += "_QR"
            with open(prefetch_info_filename) as wf:
                for i in range(len(embedding_profiles)):
                    start = self.r_size_per_table * i
                    end = self.r_size_per_table * (i+1)
                    wf.write(f"{start} {end}")

        elif self.is_TT_Rec:
            self.first_size_per_table = np.array([self.ws_translator.get_TT_Rec_size(i, self.vec_size, True, False, False) for i in range(len(embedding_profiles))])
            self.third_size_per_table = np.array([self.ws_translator.get_TT_Rec_size(i, self.vec_size, False, False, True)])
            self.reserved_page = int((np.sum(self.first_size_per_table)+np.sum(self.third_size_per_table))//self.page_offset) * self.nodes

            prefetch_info_filename += "_TT_Rec"
            with open(prefetch_info_filename) as wf:
                for i in range(len(embedding_profiles)):
                    start = np.sum(self.first_size_per_table[:i] + self.third_size_per_table[:i])
                    end = start + self.first_size_per_table[i] + self.third_size_per_table[i]
                    wf.write(f"{start} {end}")

        # preprocess for logical2physical translation
        self.table_addr_DIMM, self.table_addr_HBM = self.logical_translation(self.hot_vec_loc, self.embedding_profiles, self.vec_size, self.collisions)
        self.page_translation_HBM = [i for i in range(self.HBM_max_page_number)]
        self.page_translation_DIMM = [i for i in range(self.DIMM_max_page_number)]
        random.shuffle(self.page_translation_HBM)
        random.shuffle(self.page_translation_DIMM)

    def logical_translation(self, hot_vec_loc, embedding_profiles, vec_size, collisions):
        # logical address of Q table and 2nd core
        if self.is_QR:
            space_per_table_HBM = [0 + vec_size * (self.ws_translator.get_QR_size(i, vec_size, True)) for i in range(len(embedding_profiles))]
        elif self.is_TT_Rec:
            space_per_table_HBM = [0 + vec_size * (self.ws_translator.get_TT_Rec_size(i, vec_size, False, True, False)) for i in range(len(embedding_profiles))]

        table_addr_HBM = []
        HBM_accumulation = 0 
        HBM_accumulation += self.reserved_page
        for i in range(len(space_per_table_HBM)):
            table_addr_HBM.append(HBM_accumulation)
            HBM_accumulation += space_per_table_HBM[i]
       
        print("HBM occupied: ", HBM_accumulation/1024/1024/1024, "GB")

        return table_addr_HBM

    def physical_translation(self, table_idx, vec_idx):    
        ## R table and 1st, 3rd core is located at the front of the table

        # generate ppn
        table_logical_addr = self.table_addr_HBM[table_idx]
        if self.is_QR:
            q_idx, r_idx = self.ws_translator.get_QR_entry(self.vec_size, table_idx, vec_idx)
            q_vec_logical_addr = table_logical_addr + q_idx * self.vec_size
            r_vec_logical_addr = table_idx * self.collisions * self.vec_size + r_idx * self.vec_size

            q_vpn = int(q_vec_logical_addr // self.page_offset)
            q_ppn = self.HBM_page_translation[q_vpn]
            q_po_loc = q_vec_logical_addr % self.page_offset

            r_vpn = int(r_vec_logical_addr // self.page_offset)
            r_ppn = r_vpn
            r_po_loc = r_vec_logical_addr % self.page_offset

            q_physical_addr = int(q_ppn*self.page_offset + q_po_loc)
            r_physical_addr = int(r_ppn*self.page_offset + r_po_loc)

            return q_physical_addr, r_physical_addr
        
        elif self.is_TT_Rec:
            access = self.ws_translator.get_TT_Rec_entry(self.vec_size, table_idx, vec_idx)
            total_physical_addr = []
            for a, b, c in access:
                first_c_logical_addr = self.first_size_per_table[:table_idx] + a * self.rank * 4
                third_c_logical_addr = self.third_size_per_table[:table_idx] + c * self.rank * 4
                second_c_logical_addr = table_logical_addr + b * self.rank * self.rank * 4 * 4

                first_c_vpn, second_c_vpn, third_c_vpn = int(first_c_logical_addr // self.page_offset), int(second_c_logical_addr // self.page_offset), int(third_c_logical_addr // self.page_offset)
                first_c_ppn, second_c_ppn, third_c_ppn = first_c_vpn, self.HBM_page_translation[second_c_vpn], third_c_vpn
                first_c_po_loc, second_c_po_loc, third_c_po_loc = first_c_logical_addr % self.page_offset, second_c_logical_addr % self.page_offset, third_c_logical_addr % self.page_offset                
                first_c_physical_addr, second_c_physical_addr, third_c_physical_addr = int(first_c_ppn*self.page_offset + first_c_po_loc), int(second_c_ppn*self.page_offset + second_c_ppn), int(third_c_ppn*self.page_offset + third_c_po_loc)
                total_physical_addr.append((first_c_physical_addr, second_c_physical_addr, third_c_physical_addr))

            return total_physical_addr

# Heterogeneous memory system

# def logical_translation(self, hot_vec_loc, embedding_profiles, vec_size, collisions):        
#     if self.is_QR:
#         space_per_table_HBM = [0 + vec_size * (self.reserved_page+len(hot_vec_loc[i])) for i in range(len(hot_vec_loc))]
#         space_per_table_DIMM = [0 + vec_size * (prof_per_table.shape[0]-len(hot_vec_loc[i])) for i, prof_per_table in enumerate(embedding_profiles)]
#     elif self.is_TT_Rec:
#         space_per_table_HBM = [0 + vec_size * (self.reserved_page+collisions+len(hot_vec_loc[i])) for i in range(len(hot_vec_loc))]
#         space_per_table_DIMM = [0 + vec_size * (prof_per_table.shape[0]-len(hot_vec_loc[i])) for i, prof_per_table in enumerate(embedding_profiles)]       
    # table_addr_HBM = []
    # HBM_accumulation = 0 

    # HBM_accumulation += self.reserved_page
    # for i in range(len(space_per_table_HBM)):
    #     table_addr_HBM.append(HBM_accumulation)
    #     HBM_accumulation += space_per_table_HBM[i]
    
    # print("HBM occupied: ", HBM_accumulation/1024/1024/1024, "GB")

    # table_addr_DIMM = []
    # DIMM_accumulation = 0
    # for i in range(len(space_per_table_DIMM)):
    #     table_addr_DIMM.append(DIMM_accumulation)
    #     DIMM_accumulation += space_per_table_DIMM[i]

    # return table_addr_HBM , table_addr_DIMM