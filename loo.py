"""Module implementing gnn to annotated data"""
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bird_cloud_gnn.gnn_model import GCN
from bird_cloud_gnn.radar_dataset import RadarDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

# Filter out the UserWarning related to TypedStorage deprecation
warnings.filterwarnings("ignore", category=UserWarning, module="dgl")

MODULE_NAME = "gnn_model"
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
log_filename = os.path.join(LOG_DIR, f"{MODULE_NAME}.log")
# Initialize the logger
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# set path to data
DATA_PATH = "data/less/"
print(f"Files in the directory: {os.listdir(DATA_PATH)}")

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

# labels = np.array(dataset.labels) # do I need to provide labels?
# list of polar volumes
s = pd.Series(dataset.origin)

# initialise leave one out
# loo = LeaveOneOut()

progress_bar = tqdm(total=len(s.unique()))

for origin in s.unique():
    actual_test_idx = s[s == origin].index.to_list()
    actual_train_idx = s[s != origin].index.to_list()
    train_sampler = SubsetRandomSampler(actual_train_idx)
    test_sampler = SubsetRandomSampler(actual_test_idx)

    BATCH_SIZE = 512
    train_dataloader = GraphDataLoader(
        dataset=dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        drop_last=False,
    )
    test_dataloader = GraphDataLoader(
        dataset=dataset,
        sampler=test_sampler,
        batch_size=BATCH_SIZE,
        drop_last=False,
    )

    LEARNING_RATE = 0.01
    model = GCN(len(dataset.features), 16, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    NUM_EPOCHS = 20
    for _ in range(NUM_EPOCHS):
        model.train()
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata["x"].float())
            loss = torch.nn.functional.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            assert pred.dim() == model.num_classes

    model.eval()
    NUM_CORRECT = 0
    NUM_TESTS = 0

    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata["x"].float())
        NUM_CORRECT += (pred.argmax(1) == labels).sum().item()
        NUM_TESTS += len(labels)
        accuracy = NUM_CORRECT / NUM_TESTS
        assert pred.dim() == model.num_classes
        logging.info(f"Origin {origin} - Test accuracy: {accuracy}")

    progress_bar.set_postfix({"Origin": origin})
    progress_bar.update(1)
