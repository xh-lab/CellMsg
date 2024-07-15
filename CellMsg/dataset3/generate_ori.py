import numpy as np
import pandas as pd
from tqdm import tqdm

interaction = pd.read_csv("/home/jby2/XH/CellMsg/dataset3/ligand-receptor-interaction.csv", header=None, index_col=None)

L = interaction.iloc[1:, 0].tolist()
R = interaction.iloc[0, 1:].tolist()
print(np.array(L).shape[0])
print(np.array(R).shape[0])
interaction = interaction.iloc[1:, 1:].to_numpy()

LRP = []
num = 0

for i in tqdm(range(interaction.shape[0])):
    for j in tqdm(range(interaction.shape[1])):
        if interaction[i][j] == '1':
            num+=1
            LRP.append([L[i], R[j]])
print(num)
df = pd.DataFrame(LRP)

df.to_csv("/home/jby2/XH/CellMsg/dataset3/generation/ori_LR.csv" ,index=False, header=False)
