import pandas as pd
import numpy as np
from tqdm import tqdm

LRI_ours = pd.read_csv("/home/jby2/XH/CellMsg/dataset4/generation/LRI.csv", header=None, index_col=None)

LRI_ours = LRI_ours.drop_duplicates()

columns_to_read1 = [1, 2]
LRI_singlecellsignalr = pd.read_csv("/home/jby2/XH/CCC_compare_DB/SingleCellSignalR/human_lr.csv", usecols=columns_to_read1, header=0)
LRI_singlecellsignalr.columns = ['ligand', 'receptor']

columns_to_read2 = [0, 1]
LRI_natmi = pd.read_csv("/home/jby2/XH/CCC_compare_DB/NATMI/human_lr.csv", usecols=columns_to_read2, header=0)
LRI_natmi.columns = ['ligand', 'receptor']

columns_to_read3 = [1, 2]
LRI_cytotalk = pd.read_csv("/home/jby2/XH/CCC_compare_DB/Cytotalk/lrp_human.csv", usecols=columns_to_read3, header=0)
LRI_cytotalk.columns = ['ligand', 'receptor']

columns_to_read4 = [2, 4]
LRI_connectome = pd.read_csv("/home/jby2/XH/CCC_compare_DB/Connectome/human_lr.csv", usecols=columns_to_read4, header=0)
LRI_connectome.columns = ['ligand', 'receptor']

columns_to_read5 = [2, 3]
LRI_celltalkdb = pd.read_csv("/home/jby2/XH/CCC_compare_DB/CellTalkDB/human_lr_pair.csv", usecols=columns_to_read5, header=0)
LRI_celltalkdb.columns = ['ligand', 'receptor']

dfs = [LRI_singlecellsignalr, LRI_natmi, LRI_cytotalk, LRI_connectome, LRI_celltalkdb]
combined_df = pd.concat(dfs)
combined_df = combined_df.drop_duplicates()
print(combined_df)

print(combined_df.shape[0])
overlap_num = 0
for i in tqdm(range(LRI_ours.shape[0])):
    lo = LRI_ours.iloc[i][0]
    ro = LRI_ours.iloc[i][1] 
    for j in tqdm(range(combined_df.shape[0])):
        lc = combined_df.iloc[j]['ligand']
        rc = combined_df.iloc[j]['receptor']
        if lo == lc and ro == rc:
            overlap_num += 1
            break
            
            
total_LRI = LRI_ours.shape[0] + combined_df.shape[0] - overlap_num
with open("/home/jby2/XH/CellMsg/dataset4/CCCDB_compare/compare_fivedb.txt", mode="a") as f:
    f.write("overlap(LRI sensitivity):{}\n".format(overlap_num))
    f.write("Jaccard:{}\n".format(overlap_num/total_LRI))
    f.write("Union's number of two LRI:{}\n".format(total_LRI))