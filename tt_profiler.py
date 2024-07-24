import numpy as np
import torch

class tt_profiler:
    def __init__(self):
        self.table_idx = None
        self.emb_idx = None
        self.I = None
        self.J = None
        self.output = [[]]*3
    
    def set_I(self):
        # Use arguments in the TT-Rec Paper 
        if self.table_idx == 2:
            self.I = [200, 220, 250]
        elif self.table_idx == 11:
            self.I = [200, 200, 209]
        elif self.table_idx == 20:
            self.I = [200, 200, 200]
        elif self.table_idx == 15:
            self.I = [166, 175, 188]
        elif self.table_idx == 3:
            self.I = [125, 130, 136]
        elif self.table_idx == 23:
            self.I = [53, 72, 75]
        elif self.table_idx == 25:
            self.I = [50, 52, 55]

        # for debugging
        elif self.table_idx == -1:
            self.I = [2, 3, 4]
            self.J = [2, 2, 2]

        else:   
            pass
        
    def translate_idx(self, table_idx, emb_idx, J):
        self.table_idx = table_idx
        self.emb_idx = emb_idx
        self.J = J
        
        if table_idx in [-1, 2, 11, 20, 15, 3, 23, 25]:
            # for debugging
            if table_idx == -1:
                self.J = [2, 2, 2]
            
            self.set_I()

            if isinstance(self.emb_idx, torch.Tensor):
                self.emb_idx = self.emb_idx.cpu().numpy()

            I0 = self.I[0]
            I1 = self.I[1]
            I01 = I0 * I1

            self.output = []

            for idx in self.emb_idx:
                core_output = []

                if self.J == [2, 2, 4]:
                    # 1st-core index
                    core_output.append(np.array([0, 1]) + np.array([2, 2]) * (idx % I0))

                    # 2nd-core index
                    core_output.append(np.array([0, 1]) + np.array([2, 2]) * ((idx % I01) // I0))

                    # 3rd-core index
                    core_output.append(np.array([0, 1, 2, 3]) + np.array([4, 4, 4, 4]) * (idx // I01))

                elif self.J == [4, 4, 4]:
                    # 1st-core index
                    core_output.append(np.array([0, 1, 2, 3]) + np.array([4, 4, 4, 4]) * (idx % I0))

                    # 2nd-core index
                    core_output.append(np.array([0, 1, 2, 3]) + np.array([4, 4, 4, 4]) * ((idx % I01) // I0))

                    # 3rd-core index
                    core_output.append(np.array([0, 1, 2, 3]) + np.array([4, 4, 4, 4]) * (idx // I01))

                else:   
                    # 1st-core index
                    core_output.append(np.array([0, 1]) + np.array([2, 2]) * (idx % I0))

                    # 2nd-core index
                    core_output.append(np.array([0, 1]) + np.array([2, 2]) * ((idx % I01) // I0))

                    # 3rd-core index
                    core_output.append(np.array([0, 1]) + np.array([2, 2]) * (idx // I01))  

                self.output.append(core_output)

            # print("I:", self.I)
            # print("J:", self.J)
            print(self.output)
            # print()

        else:
            # print("No TT-table")
            # print()
            pass



# Test the class
# profiler = tt_profiler()
# profiler.translate_idx(table_idx=2, emb_idx=torch.tensor([3, 11, 3, 13], device='cuda:0'), J=[2, 2, 4])
