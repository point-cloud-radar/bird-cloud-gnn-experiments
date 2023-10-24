
import os

from bird_cloud_gnn.radar_dataset import RadarDataset

DATA_PATH = "XX"
os.listdir(DATA_PATH)

import dgl
def plot_graph(G, spatial_columns=(0,1)):
    assert isinstance(G , dgl.DGLGraph)
    pp=dgl.to_networkx(G,['x'], ['a'])
    for i in range(pp.number_of_nodes()):
        pp.nodes[i]['pos'] = (pp.nodes[i]['x'].numpy()[spatial_columns[0]],pp.nodes[i]['x'].numpy()[spatial_columns[1]])
    pos=nx.get_node_attributes(pp,'pos')
    nx.draw(pp, pos,  with_labels=True, connectionstyle='arc3, rad = 0.1')

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
med=850
aetp=False
i=5270
for med in [3,350,650,850]:
    for aetp in [True,False]:
        dataset = RadarDataset(
           data=DATA_PATH,
            features=['x','y'],
            target="BIOLOGY",
            num_nodes=50,
            max_poi_per_label=50,
            max_edge_distance=med,
            add_edges_to_poi=aetp,
        )
        for i in np.random.choice(np.array(range(len(dataset))),10, False):
            plt.clf()
            plt.figure(figsize=(8,8))
            plot_graph(dataset[i][0])
            plt.gca().set_aspect('equal')    
            plt.savefig(f"graphs_aetp_{aetp}_med_{med}_grp_{i:07d}.png")
            plt.close()

if False:
    for i in np.random.choice(np.array(range(len(dataset))),10, False):
        plot_graph(dataset[i][0])
        plt.show()


