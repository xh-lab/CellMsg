# CellMsg: graph convolutional networks for ligand-receptor-mediated cell-cell communication analysis
=========================================================================================


[![license](https://img.shields.io/badge/python_-3.11.7_-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-2.0.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/scanpy_-1.10.1_-red)](https://scanpy.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/anndata_-0.10.8_-green)](https://anndata-tutorials.readthedocs.io/en/latest/index.html/)
[![license](https://img.shields.io/badge/LIANA+_-yellow)](https://github.com/saezlab/liana)
[![license](https://img.shields.io/badge/iFeature_-yellow)](https://github.com/Superzchen/iFeature/)


CellMsg is a method for analyzing cell-cell communication mediated by ligand-receptor interactions. (i) The CellMsg method is a framework that accurately captures potential ligand-receptor interactions, thereby effectively inferring cell-cell communications (CCCs). (ii) The CellMsg method organizes ligand-receptor pairs into an adjacency matrix format, obtains protein features using iFeature, and performs feature extraction through GCNConv. This is followed by binary classification tasks using linear layers. Multiple layers of GCNConv with skip connections are added to ensure comprehensive node neighborhood information, avoiding issues like over-smoothing and gradient vanishing, thereby improving model accuracy. The overview figure of CellMsg is shown as follows.


![Image text](https://github.com/xh-lab/CellMsg/blob/main/cellmsgworkflow.png)


The overview of CellMsg.  a. Ligand-Receptor Interaction Prediction, including Data Preprocessing and Ligand-Receptor Interaction Classification. The data preprocessing section extracts multimodal features of ligands and receptors to construct an initial feature matrix and constructs an adjacency matrix based on known associations of ligands and receptors. The Ligand-Receptor Interaction Classification section uses these matrices as inputs to the graph convolutional networks to classify LRIs. b. Cell-Cell Communication Inference, including Ligand-Receptor Interaction Screening and Cell-Cell Communication Strength Measurement. The ligand-receptor interaction screening section filters high-confidence LRIs and then calculates the thresholding result, product result, and cell result for each scRNA-seq data. On this basis, the cell-cell communication strength measurement section uses a three-point estimation method to calculate the CCC strength between different cell types. c. Cell-Cell Communication Visualization, including the communication heatmap between different cell types, the communication network between different cell types, and the communication heatmap of the most active LR pairs between different cell types.

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

**Notes**: Due to the large size of some datas, we uploaded them to the Google Drive, if some files cannot be found, please look for them [here](https://drive.google.com/drive/u/0/folders/1IX3JMGCfD8bQNL9vDmEjbE3Mk0cNe9Z9). 

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
| ligand-receptor-interaction.csv | This file contains information about ligand-receptor interactions. |
| final_model.pth (stored in google drive) | The final model for predicting LRIs. |
| LRI_predicted.csv | LRIs that predicted by CellMsg. |
| original_LRI.csv, LRI_ori_.csv, origin_LRI.csv | LRIs that we collected. |

## 1, acquire feature file from sequence file using iFeature
**Notes**: Since the processing steps for all sequence files are identical, we will proceed to process one of the sequence files.
```
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type AAC --out ligand_aac.csv
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type CKSAAP --out ligand_cksaap.csv
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type CTriad --out ligand_ctriad.csv
python iFeature.py --file CellMsg/dataset1/ligand_sequence.txt --type PAAC --out ligand_paac.csv
Then, the four features were merged to generate the final feature file for all ligands, where each row represents the features of one ligand, with the number of rows equating to the number of ligands.
```

## 2, training an LRI prediction model
**Notes**: Since the steps for training the LRI prediction model are the same for all datasets, let's proceed with processing Dataset 1.
```
python CellMsg/dataset1/CellMsg.py
```
## 3, preditcing LRIs
**Notes**: Since the steps for predicting LRIs are the same for all datasets, let's proceed with processing Dataset 1.
```
python CellMsg/dataset1/generate_lr.py
python CellMsg/dataset1/ensp_to_gname.py
```
Through the above steps, we obtained predicted LRIs with high confidence, which are then merged with the LRIs previously collected to serve as LRIs identified by CellMsg.

## 4, Cell-Cell Communication Strength Measurement
```
python CellMsg/CCC_Analysis/Processing_scRNA-seq_data.py
python CellMsg/CCC_Analysis/The_three-point_estimation_method.py
```
Through the above steps, we obtained the cell-cell communication strength matrix processed using the three-point evaluation method, and we generated the cell communication heatmap and cell communication network.

## 5, Visualization analysis of cell-cell communication
```
python CellMsg/CCC_Analysis/The_number_of_LRIs.py
```
Through the steps outlined above, we have obtained Three_LRi_num.pdf and Three_LRi_num.csv, which show the number of LRIs mediating communication between cell types in human melanoma tissues.

```
python CellMsg/CCC_Analysis/Top.py
```
Through the above command, we obtained Top.pdf and Top_data.csv, which display the three most likely LR pairs mediating communication between melanoma cancer cells and six other cell types.

=========================================================================================



# Contributing 
All authors were involved in the conceptualization of the SpaCCC method. BYJ and SLP conceived and supervised the project. BYJ and HX collected the data. HX completed the coding for the project. BYJ, HX and SLP contributed to the review of the manuscript before submission for publication. All authors read and approved the final manuscript.

# cite
<p align="center">
  <a href="https://clustrmaps.com/site/1bpq2">
     <img width="200"  src="https://clustrmaps.com/map_v2.png?cl=ffffff&w=268&t=m&d=4hIDPHzBcvyZcFn8iDMpEM-PyYTzzqGtngzRP7_HkNs" />
   </a>
</p>

<p align="center">
  <a href="#">
     <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fxh-lab%2FCellMsg&labelColor=%233499cc&countColor=%2370c168" />
   </a>
</p>


# Contacts
If you have any questions or comments, please feel free to email: Shaoliang Peng (slpeng@hnu.edu.cn); (Boya Ji) byj@hnu.edu.cn; (Hong Xia) hongxia@hnu.edu.cn.

# License

[MIT &copy Richard McRichface.](../LICENSE)
