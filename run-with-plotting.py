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

torch.manual_seed(42)

DATA_PATH = "/opt/bird-data/"
os.listdir(DATA_PATH)

features = [
    "range",
    "azimuth",
    "elevation",
    "z",
    "DBZH",
    "TH",
    "VRADH",
    "RHOHV",
    "PHIDP",
    "centered_x",
    "centered_y",
]
MAX_EDGE_DISTANCE=650.0
NUM_NODES=50
LEARNING_RATE=0.001
SEED=42
BATCH_SIZE=512
dataset = RadarDataset(
    data=DATA_PATH,
    features=features,
    target="BIOLOGY",
    num_nodes=NUM_NODES,
    max_poi_per_label=500,
    max_edge_distance=MAX_EDGE_DISTANCE,
)

print(f"Dataset size: {len(dataset)}")

num_examples = len(dataset)
num_train = int(num_examples * 0.8)
np.random.seed(SEED)
train_idx = np.random.choice(num_examples, num_train, replace=False)
test_idx = np.setdiff1d(np.arange(0, num_examples), train_idx, assume_unique=True)

model = GCN(len(dataset.features), 16, 2)

train_dataloader, test_dataloader = get_dataloaders(
    dataset, train_idx, test_idx, batch_size=BATCH_SIZE
)

callback = CombinedCallback([
    TensorboardCallback(
        log_dir = "/".join([
            "runs",
            dataset.oneline_description(),
            model.oneline_description(),
            f"LR_{LEARNING_RATE}-BS_{BATCH_SIZE}-SEED{SEED}",
          ]
        ),
    ),
    EarlyStopperCallback(patience=50),
])

model.fit_and_evaluate(
    train_dataloader,
    test_dataloader,
    callback,
    learning_rate=LEARNING_RATE,
    num_epochs=500,
)
