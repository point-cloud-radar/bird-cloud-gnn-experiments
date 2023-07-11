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
from pytorch import nn

torch.manual_seed(46)

DATA_PATH = "../data/volume_2/parquet/"
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
MAX_EDGE_DISTANCE = 650.0
NUM_NODES = 50
LEARNING_RATE = 0.005
SEED = 46
BATCH_SIZE = 512
dataset = RadarDataset(
    data=DATA_PATH,
    features=features,
    target="BIOLOGY",
    num_nodes=NUM_NODES,
    max_poi_per_label=5000,  # 500
    max_edge_distance=MAX_EDGE_DISTANCE,
    use_missing_indicator_columns=True,
    add_edges_to_poi=True,
)
print(f"Dataset size: {len(dataset)}")

num_examples = len(dataset)
num_train = int(num_examples * 0.8)
np.random.seed(SEED)
train_idx = np.random.choice(num_examples, num_train, replace=False)
test_idx = np.setdiff1d(np.arange(0, num_examples), train_idx, assume_unique=True)

model = GCN(len(dataset.features), [(16, nn.ReLU()), (16, nn.ReLU()), (2, None)])

train_dataloader, test_dataloader = get_dataloaders(
    dataset, train_idx, test_idx, batch_size=BATCH_SIZE
)

callback = CombinedCallback(
    [
        TensorboardCallback(
            log_dir="/".join(
                [
                    "runs",
                    dataset.oneline_description(),
                    model.oneline_description(),
                    f"LR_{LEARNING_RATE}-BS_{BATCH_SIZE}-SEED{SEED}",
                    f"1,[150,300],0.5",
                ]
            ),
        ),
        EarlyStopperCallback(patience=500),
    ]
)

model.fit_and_evaluate(
    train_dataloader,
    test_dataloader,
    callback,
    learning_rate=LEARNING_RATE,
    num_epochs=500,
    sch_explr_gamma=1,
    sch_multisteplr_milestones=[150, 300],
    sch_multisteplr_gamma=0.5,
)
