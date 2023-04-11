# %%
import os

from bird_cloud_gnn.radar_dataset import RadarDataset
from functions import show_one_graph

datapath = os.path.join(os.path.dirname(__file__), "..", "data", "small")
dataset = RadarDataset(
    datapath,
    ["x", "y", "z"],
    "BIOLOGY",
    min_neighbours=50,
    max_distance=5000.0,
    max_edge_distance=200.0,
)

#%%
graph_index = len(dataset) - 1
G = dataset[graph_index][0]
show_one_graph(G, G.ndata["x"])

# %%
