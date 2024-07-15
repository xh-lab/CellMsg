# CellMsg: graph convolutional networks for ligand-receptor-mediated cell-cell communication analysis
=========================================================================================


[![license](https://img.shields.io/badge/python_-3.11.7_-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-2.0.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/scanpy_-1.10.1_-red)](https://scanpy.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/anndata_-0.10.8_-green)](https://anndata-tutorials.readthedocs.io/en/latest/index.html/)
[![license](https://img.shields.io/badge/LIANA+_-yellow)](https://github.com/saezlab/liana)
[![license](https://img.shields.io/badge/iFeature_-yellow)](https://github.com/Superzchen/iFeature/)


CellMsg is a method for analyzing cell-cell communication mediated by ligand-receptor interactions. (i) The CellMsg method is a framework that accurately captures potential ligand-receptor interactions, thereby effectively inferring cell-cell communications (CCCs). (ii) The CellMsg method organizes ligand-receptor pairs into an adjacency matrix format, obtains protein features using iFeature, and performs feature extraction through GCNConv. This is followed by binary classification tasks using linear layers. Multiple layers of GCNConv with skip connections are added to ensure comprehensive node neighborhood information, avoiding issues like over-smoothing and gradient vanishing, thereby improving model accuracy. The overview figure of CellMsg is shown as follows.


![Image text](https://github.com/xh-lab/CellMsg/blob/main/workflow.pdf)


The overview of CellMsg. (a) Ligand-Receptor Interaction Prediction, including Data Preprocessing and Ligand-Receptor Interaction Classification. The data preprocessing section uses iFeature to extract protein features to construct the protein feature matrix and organize LR pairs into the LR interaction matrix. These matrices are then used as inputs to the LRI prediction model to generate LRIs. (b) Cell-Cell Communication Inference, including Ligand-Receptor Interaction Screening and Cell-Cell Communication Strength Measurement. The ligand-receptor interaction screening section filters high-confidence LRIs in scRNA-seq data, then calculates the thresholding result, product result, and cell result using the filtered LRIs. Finally, the three-point estimation method is applied to the three matrices to derive the final CCC strength. (c) Cell-Cell Communication Visualization, generating a cell-cell communication heatmap, cell-cell communication network, and visualization of the most active LR pairs in each cell type pair communication from the cell-cell communication strength matrix.
## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Cite](#cite)
- [Contacts](#contacts)
- [License](#license)


## Installation
CellMsg is tested to work under:

```
* Anaconda 24.1.2
* Python 3.11.7
* Torch 2.0.1
* Scanpy 1.10.1
* Anndata 0.10.8
* R 4.2.2
* Numpy 1.24.4
* iFeature
* Other basic python and r toolkits
```
### Installation of other dependencies
* Install [LIANA+](https://github.com/saezlab/liana-py/) using pip install liana if you encounter any issue.
* Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/) using pip install torch_geometric if you encounter any issue.
* Install [seaborn-0.13.2](https://seaborn.pydata.org/) using pip install seaborn if you encounter any issue.
* Install [networkx-2.8.8](https://pypi.org/project/networkx/2.8.8/) using pip install networkx if you encounter any issue.


# Quick start
To reproduce our results:

**Notes**: Due to the large size of some datas, we uploaded them to the Google Drive, if some files cannot be found, please look for them [here](https://drive.google.com/drive/u/0/my-drive). 

## Data Description
| **File name** | **Description** |
| :---: | :--- |
| CellTalkDB/human_lr_pair.rds and CellTalkDB/mouse_lr_pair.rds | The human and mouse LRI databases provided by CellTalkDB can be obtained from https://github.com/ZJUFanLab/CellTalkDB/tree/master/database. |
| CellTalkDB/human_lr_pair.csv and CellTalkDB/mouse_lr_pair.csv | The CSV files converted from CellTalkDB/human_lr_pair.rds and CellTalkDB/mouse_lr_pair.rds using R. |
| Connectome/ncomms8866_human.rda and Connectome/ncomms8866_mouse.rda | The human and mouse LRI databases provided by Connectome can be obtained from https://github.com/msraredon/Connectome/tree/master/data. |
| Connectome/human_lr.csv and Connectome/mouse_lr.csv | The CSV files converted from Connectome/ncomms8866_human.rda and Connectome/ncomms8866_mouse.rda using R. |
| Cytotalk/lrp_human.rda and Cytotalk/lrp_mouse.rda | The human and mouse LRI databases provided by Cytotalk can be obtained from https://github.com/tanlabcode/CytoTalk/tree/master/data. |
| Cytotalk/lrp_human.csv and Cytotalk/lrp_mouse.csv | The CSV files converted from Cytotalk/lrp_human.rda and Cytotalk/lrp_mouse.rda using R. |
| NATMI/human_lr.csv | The human LRI database provided by NATMI can be obtained from https://github.com/forrest-lab/NATMI/blob/master/lrdbs/lrc2p.csv. |
| SingleCellSignalR/LRdb.rda | The human LRI database provided by SingleCellSignalR can be obtained from https://github.com/SCA-IRCM/SingleCellSignalR/tree/master/data. |
| SingleCellSignalR/human_lr.csv | The CSV files converted from SingleCellSignalR/LRdb.rda using R. |
| LRI.csv | The LR pairs identified by CellMsg. | 
| cell2ct.csv and cell2ct.txt | Cell annotation files, including mappings from cells to cell types (both represented numerically). |
| mart_export.txt, uniprotid2gn.txt, ensmusg.txt and ensmusp.txt | Mapping files of protein identifiers to gene names. |
| ligand_sequence.txt and receptor_sequence.txt | ligand and receptor sequence files, they serve as input files for iFeature to generate corresponding ligand or receptor features. |
| ligand_res_fea.csv and receptor_res_fea.csv (stored in google drive) | ligand feature and receptor feature files obtained after processing with iFeature. | 

## 1, acquire feature file from sequence file using iFeature
**Notes**: Since the processing steps for all sequence files are identical, we will proceed to process one of the sequence files.
```
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type AAC --out ligand_aac.csv
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type CKSAAP --out ligand_cksaap.csv
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type CTriad --out ligand_ctriad.csv
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type PAAC --out ligand_paac.csv
Then, the four features were merged to generate the final feature file for all ligands, where each row represents the features of one ligand, with the number of rows equating to the number of ligands.
```
