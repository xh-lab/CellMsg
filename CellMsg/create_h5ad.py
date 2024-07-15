import pandas as pd
import anndata
import scanpy as sc
import os


df = pd.read_csv("/home/jby2/XH/network_val/gene_exp.csv", header=None, index_col=None)
print(df.shape)

df_transposed = df.T
df_transposed.to_csv("/home/jby2/XH/network_val/transpose.csv", index=False, header=False)


exp = pd.read_csv('/home/jby2/XH/network_val/transpose.csv', index_col=0, header=0)
meta_data = pd.read_csv("/home/jby2/XH/network_val/cell2ct.csv", index_col=0, header=0) 

adata = sc.AnnData(X=exp.values, obs=meta_data, var=pd.DataFrame(index=exp.columns))

output_dir = '/home/jby2/XH/network_val/'
os.makedirs(output_dir, exist_ok=True)


output_file = os.path.join(output_dir, 'data.h5ad')


adata.write(output_file)

t = sc.read_h5ad('/home/jby2/XH/network_val/data.h5ad')
print(t.X)
print("##########################################################")
print(t.obs)
print("##########################################################")
print(t.obs["cell_type"])
print("##########################################################")
print(t.var)


#adata = anndata.AnnData(X=adata)
#adata.obs_names_make_unique()
#adata.write_h5ad('/home/jby2/XH/CellMsg/test.h5ad')



#t = sc.read_h5ad("/home/jby2/XH/CellMsg/test.h5ad")
#print(t.to_df())
#print(t.obs)

#df = pd.read_csv('/home/jby2/XH/CellMsg/transpose.csv', header=None)
#print(df.shape)
#cell_type = df.iloc[:, 0].iloc[1:].tolist()
#cell_type = np.array(cell_type).reshape(-1, 1)
#print(cell_type.shape)
#print(cell_type)

#adata.obs['cell_type'] = pd.Series(cell_type, index=adata.obs.index)
#print(adata.obs)