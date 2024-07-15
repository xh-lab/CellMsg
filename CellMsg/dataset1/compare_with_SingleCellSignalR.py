import pandas as pd
import numpy as np
from tqdm import tqdm

LRI_ours = pd.read_csv("/home/jby2/XH/CellMsg/dataset1/LRI.csv", header=None, index_col=None)
columns_to_read = [1, 2]
LRI_singlecellsignalr = pd.read_csv("/home/jby2/XH/CCC_compare_DB/SingleCellSignalR/human_lr.csv", usecols=columns_to_read, header=0)
print(LRI_singlecellsignalr.shape[0])
overlap_num = 0
for i in tqdm(range(LRI_ours.shape[0])):
    lo = LRI_ours.iloc[i][0]
    ro = LRI_ours.iloc[i][1] 
    for j in tqdm(range(LRI_singlecellsignalr.shape[0])):
        lc = LRI_singlecellsignalr.iloc[j]['ligand']
        rc = LRI_singlecellsignalr.iloc[j]['receptor']
        if lo == lc and ro == rc:
            overlap_num += 1
            break
            
            
total_LRI = LRI_ours.shape[0] + LRI_singlecellsignalr.shape[0] - overlap_num
with open("/home/jby2/XH/CellMsg/dataset1/CCCDB_compare/compare_singlecellsignalr.txt", mode="a") as f:
    f.write("overlap(LRI sensitivity):{}\n".format(overlap_num))
    f.write("Jaccard:{}\n".format(overlap_num/total_LRI))
    f.write("Union's number of two LRI:{}\n".format(total_LRI))