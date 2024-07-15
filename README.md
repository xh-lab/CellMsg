# CellMsg: graph convolutional networks for ligand-receptor-mediated cell-cell communication analysis
=========================================================================================


[![license](https://img.shields.io/badge/python_-3.11.7_-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-2.0.1_-orange)](https://pytorch.org/)
[![license](https://img.shields.io/badge/scanpy_-1.10.1_-red)](https://scanpy.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/anndata_-0.10.8_-green)](https://anndata-tutorials.readthedocs.io/en/latest/index.html/)
[![license](https://img.shields.io/badge/liana_-yellow)](https://github.com/saezlab/liana)

CellMsg is a method for analyzing cell-cell communication mediated by ligand-receptor interactions. (i) The CellMsg method is a framework that accurately captures potential ligand-receptor interactions, thereby effectively inferring cell-cell communications (CCCs). (ii) The CellMsg method organizes ligand-receptor pairs into an adjacency matrix format, obtains protein features using iFeature, and performs feature extraction through GCNConv. This is followed by binary classification tasks using linear layers. Multiple layers of GCNConv with skip connections are added to ensure comprehensive node neighborhood information, avoiding issues like oversmoothing and gradient vanishing, thereby improving model accuracy. The overview figure of CellMsg is shown as follows.


![Image text](https://github.com/xh-lab/CellMsg/blob/main/workflow.pdf)



