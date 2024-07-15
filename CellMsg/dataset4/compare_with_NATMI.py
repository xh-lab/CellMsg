import pandas as pd
import numpy as np
from tqdm import tqdm

LRI_ours = pd.read_csv("/home/jby2/XH/CellMsg/dataset4/generation/LRI.csv", header=None, index_col=None)

LRI_ours = LRI_ours.drop_duplicates()

columns_to_read = [0, 1]
LRI_natmi = pd.read_csv("/home/jby2/XH/CCC_compare_DB/NATMI/human_lr.csv", usecols=columns_to_read, header=0)
print(LRI_natmi.shape[0])
overlap_num = 0
for i in tqdm(range(LRI_ours.shape[0])):
    lo = LRI_ours.iloc[i][0]
    ro = LRI_ours.iloc[i][1] 
    for j in tqdm(range(LRI_natmi.shape[0])):
        lc = LRI_natmi.iloc[j]['Ligand gene symbol']
        rc = LRI_natmi.iloc[j]['Receptor gene symbol']
        if lo == lc and ro == rc:
            overlap_num += 1
            break
            
            
total_LRI = LRI_ours.shape[0] + LRI_natmi.shape[0] - overlap_num
with open("/home/jby2/XH/CellMsg/dataset4/CCCDB_compare/compare_natmi.txt", mode="a") as f:
    f.write("overlap(LRI sensitivity):{}\n".format(overlap_num))
    f.write("Jaccard:{}\n".format(overlap_num/total_LRI))
    f.write("Union's number of two LRI:{}\n".format(total_LRI))