"""run file
do not track with git
"""

import os

import torch
from bird_cloud_gnn.cross_validation import (
    get_dataloaders,
    kfold_evaluate,
    leave_one_origin_out_evaluate,
)
from bird_cloud_gnn.gnn_model import GCN
from bird_cloud_gnn.radar_dataset import RadarDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

DATA_PATH = "data/less/"
os.listdir(DATA_PATH)

features = [
    "range",
    "azimuth",
    "elevation",
    "x",
    "y",
    "z",
    "DBZH",
    "DBZV",
    "TH",
    "TV",
    "PHIDP",
    "RHOHV",
]

dataset = RadarDataset(
    data=DATA_PATH,
    features=features,
    target="BIOLOGY",
    num_nodes=20,
    max_poi_per_label=2000,
    max_edge_distance=5_000.0,
)


num_examples = len(dataset)
num_train = int(num_examples * 0.8)
train_idx = torch.arange(num_train)
test_idx = torch.arange(num_train, num_examples)

model = GCN(len(dataset.features), 16, 2)

train_dataloader, test_dataloader = get_dataloaders(
    dataset, train_idx, test_idx, batch_size=512
)

model.fit(train_dataloader)
model.evaluate(test_dataloader)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor])


kfold_evaluate(dataset)
leave_one_origin_out_evaluate(dataset)
