import pandas as pd
import numpy as np
from tqdm import tqdm

LRI_ours = pd.read_csv("/home/jby2/XH/CellMsg/dataset2/generation/LRI.csv", header=None, index_col=None)
columns_to_read = [1, 2]
LRI_cytotalk = pd.read_csv("/home/jby2/XH/CCC_compare_DB/Cytotalk/lrp_mouse.csv", usecols=columns_to_read, header=0)
print(LRI_cytotalk.shape[0])
overlap_num = 0
for i in tqdm(range(LRI_ours.shape[0])):
    lo = LRI_ours.iloc[i][0]
    ro = LRI_ours.iloc[i][1] 
    for j in tqdm(range(LRI_cytotalk.shape[0])):
        lc = LRI_cytotalk.iloc[j]['ligand']
        rc = LRI_cytotalk.iloc[j]['receptor']
        if lo == lc and ro == rc:
            overlap_num += 1
            break
            
            
total_LRI = LRI_ours.shape[0] + LRI_cytotalk.shape[0] - overlap_num
with open("/home/jby2/XH/CellMsg/dataset2/CCCDB_compare/compare_cytotalk.txt", mode="a") as f:
    f.write("overlap(LRI sensitivity):{}\n".format(overlap_num))
    f.write("Jaccard:{}\n".format(overlap_num/total_LRI))
    f.write("Union's number of two LRI:{}\n".format(total_LRI))