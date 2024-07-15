import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from scipy.sparse import csr_matrix
from torch_geometric.nn.conv import GCNConv
from tqdm import tqdm


interaction_ = pd.read_csv("/home/jby2/XH/CellMsg/dataset4/ligand-receptor-interaction.csv", header=None, index_col=None).to_numpy()
proteinFeature_L = pd.read_csv("/home/jby2/XH/CellMsg/dataset4/ligand_res_fea.csv", header=None, index_col=None)
proteinFeature_R = pd.read_csv("/home/jby2/XH/CellMsg/dataset4/receptor_res_fea.csv", header=None, index_col=None)

proteinFeature_L = proteinFeature_L.drop(columns=proteinFeature_L.columns[0]).to_numpy()
proteinFeature_R = proteinFeature_R.drop(columns=proteinFeature_R.columns[0]).to_numpy()
L_R_fea = np.vstack((proteinFeature_L, proteinFeature_R))
L_R_fea = torch.tensor(L_R_fea, dtype=torch.float32)

interaction = pd.read_csv("/home/jby2/XH/CellMsg/dataset4/ligand-receptor-interaction.csv", header=None, index_col=None, skiprows=1)
interaction = interaction.drop(columns=interaction.columns[0]).to_numpy()
whole_interaction = np.zeros((interaction.shape[0] + interaction.shape[1], interaction.shape[0] + interaction.shape[1]))
whole_interaction[:interaction.shape[0], interaction.shape[0]:] = interaction
interaction1 = csr_matrix(whole_interaction)
row, col = interaction1.nonzero()
edge_index = np.vstack([row, col])
edge_index = torch.from_numpy(edge_index)



# acquire receptor sequence
receptor_seq = np.delete(interaction_[0], 0)
receptor_seq = receptor_seq.reshape(-1, 1)
# acquire ligand sequence
temp = interaction_[1:]
ligand_seq = temp[: ,[0]]

print(receptor_seq.shape)
print(ligand_seq.shape)



def generate_sample(interaction):
    sample = []
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            if int(interaction[i][j]) == 0:
                sample.append((i, j))
    return sample
    
sample = generate_sample(interaction)
#print(type(feature))
#print(type(position))
#print(np.array(feature).shape)
#print(np.array(position).shape)


# define similar model as the CellMsg.py
class GCN_MLP(torch.nn.Module):
    def __init__(self):
        super(GCN_MLP, self).__init__()
        self.conv1 = GCNConv(2813, 1024)
        self.conv2 = GCNConv(1024, 400)
        #self.conv3 = GCNConv(400, 400)
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.f1 = nn.Linear(2813, 1024)
        self.f2 = nn.Linear(1024, 400)

    def forward(self, x, a, sample_train):    
        r1 = self.f1(x)    
        x = self.conv1(x, a)
        x = r1 + x
        x = f.relu(x)
        
        r2 = self.f2(x)
        x = self.conv2(x, a)  
        x = x + r2
        x = f.relu(x)
        
        #r3 = x
        #x = self.conv3(x, a)
        #x = x + r3
        #x = f.relu(x)
        
        
        l_embedding = x[:interaction.shape[0], :]
        r_embedding = x[interaction.shape[0]:, :]
        feature_train = []
        
        for i in sample_train:
            l_temp = l_embedding[i[0]]
            r_temp = r_embedding[i[1]]
            temp = np.append(l_temp.detach().numpy(), r_temp.detach().numpy())
            feature_train.append(temp)
        
        x = torch.tensor(feature_train, dtype=torch.float32)
        
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.fc3(x)
        x = f.sigmoid(x)
        return x
        

model = GCN_MLP()
model.load_state_dict(torch.load("/home/jby2/XH/CellMsg/dataset4/final_model.pth"))


model.eval()
num = 0 
with torch.no_grad(): 
# predicting
    outputs = model(L_R_fea, edge_index, sample)
    for i, j in tqdm(enumerate(outputs), total=outputs.shape[0]):
        if j >= 0.999:
            with open('/home/jby2/XH/CellMsg/dataset4/generation/confidence_lr_ens.csv', 'a') as file:
                file.write(f"{ligand_seq[sample[i][0]][0]}, {receptor_seq[sample[i][1]][0]}\n")  # acquire LRI
            num += 1

print(num)