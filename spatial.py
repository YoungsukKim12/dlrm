# numpy
import numpy as np
from numpy import random as ra
import matplotlib.pyplot as plt

# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler

#Original parameter
ps_data_pt = "/mnt/storage/sjm/kys_dlrm/dlrm/input/"
total_day = 7
criteo_kaggle_path = [ps_data_pt + "kaggleAdDisplayChallenge_processed.npz"]


class CriteoKaggleDataset(Dataset):
    def __init__(
        self,
        ln_emb,
        criteo_kaggle_path
        ):
        super().__init__
        self.ln_emb = ln_emb
        self.criteo_kaggle_path = criteo_kaggle_path
    
    def __getitem__(self, idx):
        f = np.load(self.criteo_kaggle_path[idx])
        data = torch.from_numpy(np.transpose(f["X_cat"]))
        return data

    def __len__(self):
        return len(self.criteo_kaggle_path)



def kaggle_run():
    ln_emb = (np.load(ps_data_pt + "train_fea_count.npz"))
    ln_emb = ln_emb["counts"]
    kaggle_loader = CriteoKaggleDataset(ln_emb, criteo_kaggle_path)
    cnt_q_table = torch.zeros(len(ln_emb))
    cnt_r_table = torch.zeros(len(ln_emb))
    mini_batch_size = 128
    hash_collision_l = [4,8,16,32,64]
    p_l = range(2, 64) #R vector는 하나의 
    plot_q = []
    plot_r = []
    hash_collision = 4
    sparse_idx = next(iter(kaggle_loader))
    for p in p_l:
        query_cnt = 0
        cnt_q_table = torch.zeros(len(ln_emb))
        cnt_r_table = torch.zeros(len(ln_emb))
        query_total = int(sparse_idx.shape[0]/mini_batch_size) + 1
        query_cnt += query_total
        for l, idx in enumerate(sparse_idx):
            for i in range(query_total):
                query = idx[i*mini_batch_size:(i+1)*mini_batch_size]
                q_query, r_query = np.divmod(query, hash_collision)
                cnt_q = len(torch.unique(torch.tensor((q_query//p).tolist())))/((ln_emb[l]//hash_collision)//p+1) #normalize
                cnt_r = len(torch.unique(torch.tensor((r_query//p).tolist())))/((hash_collision-1)//p+1)    #normalize
                cnt_q_table[l] += cnt_q
                cnt_r_table[l] += cnt_r
                if cnt_q > 1:
                    print("cnt_q:{}".format(cnt_q))
                if cnt_r > 1:
                    print("cnt_r:{}".format(cnt_r))
        cnt_table = np.stack((cnt_q_table, cnt_r_table), axis=1)        
        cnt_day = cnt_table/query_cnt
        plot_q.append(np.mean(cnt_day[:,0]))
        plot_r.append(np.mean(cnt_day[:,1]))
    plt.figure()
    plt.plot(p_l,plot_q, label="Q table")
    plt.plot(p_l,plot_r, label="R table")
    plt.xlabel("Portion")
    plt.ylabel("Spatial Locality")
    plt.legend()
    plt.savefig("Q_R_col_{}".format(hash_collision))
    plt.show()
if __name__=="__main__":
    kaggle_run()