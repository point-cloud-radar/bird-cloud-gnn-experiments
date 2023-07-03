import os

import torch
from bird_cloud_gnn.cross_validation import (
    get_dataloaders,
)
from bird_cloud_gnn.callback import (
    TensorboardCallback,
    EarlyStopperCallback,
    CombinedCallback,
)
from bird_cloud_gnn.gnn_model import GCN
from bird_cloud_gnn.radar_dataset import RadarDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

DATA_PATH = "/opt/bird-data/"
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
    num_neighbours=50,
    max_poi_per_label=500,
    max_edge_distance=650.0,
)

print(f"Dataset size: {len(dataset)}")

num_examples = len(dataset)
num_train = int(num_examples * 0.8)
np.random.seed(42)
train_idx = np.random.choice(num_examples, num_train, replace=False)
test_idx = np.setdiff1d(np.arange(0, num_examples), train_idx, assume_unique=True)

model = GCN(len(dataset.features), 16, 2)

train_dataloader, test_dataloader = get_dataloaders(
    dataset, train_idx, test_idx, batch_size=512
)

callback = CombinedCallback([
    TensorboardCallback(),
    EarlyStopperCallback(patience=50),
])

model.fit_and_evaluate(
    train_dataloader,
    test_dataloader,
    callback,
    num_epochs=500,
)
