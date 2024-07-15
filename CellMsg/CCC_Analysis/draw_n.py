import pandas as pd
import numpy as np
import seaborn as sns
from time import time
import networkx as nx
import matplotlib.pyplot as plt
import os
import psutil
import textwrap


cancer = r'/home/jby2/XH/CellMsg/CCC_Analysis'  # File directory for cancer species
x_ytick = ['Melanoma cancer cells', 'T cells', 'B cells', 'Macrophages', 'Endothelial cells', 'CAFs', 'NK cells']  # Melanoma->0...NK->6

fig = plt.figure()
df = pd.read_csv(cancer + "/case_study.csv", index_col=0)
matrix = df.values.tolist()
a = df.shape[0]
node_colors = ['pink', 'orange', 'green', 'y', 'm', 'c', 'gray', 'violet', 'r', 'khaki', 'darkred', 'saddlebrown'][:a]
G = nx.DiGraph()
for i, name in enumerate(df.index):
    G.add_node(name, node_color=node_colors[i % len(node_colors)])
for i, row in enumerate(matrix):
    for j, weight in enumerate(row):
        if weight != 0:
            G.add_edge(df.index[i], df.columns[j], weight=weight, edge_color=node_colors[i % len(node_colors)])
pos = nx.circular_layout(G)
edge_widths = [6 * d['weight'] for u, v, d in G.edges(data=True)]
edge_colors = [d['edge_color'] for u, v, d in G.edges(data=True)]
node_colors = [d['node_color'] for n, d in G.nodes(data=True)]

# Wrap labels to a maximum width
max_label_width = 10  # Set the maximum width for labels
labels = {n: "\n".join(textwrap.wrap(n, max_label_width)) for n in G.nodes()}

nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.8, edge_color=edge_colors, arrowsize=14, node_size=300, connectionstyle='arc3,rad=0.1')
nx.draw_networkx_nodes(G, pos, node_shape='o', node_color=node_colors, node_size=250)
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='sans-serif')
plt.axis('off')
plt.savefig(cancer + '/case_study_networkx.pdf', dpi=1080, bbox_inches='tight')