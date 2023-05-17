"""Module implementing gnn to annotated data"""
import logging
import os
import warnings

import numpy as np
import torch
from bird_cloud_gnn.gnn_model import GCN
from bird_cloud_gnn.radar_dataset import RadarDataset
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import StratifiedKFold
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
logging.basicConfig(filename=log_filename, level=logging.INFO)

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
    data_folder=DATA_PATH,
    features=features,
    target="BIOLOGY",
    max_distance=500.0,  # 1500
    min_neighbours=100,
    max_edge_distance=50.0,  # 355
)

labels = np.array(dataset.labels)

# highly imballanced dataset
# 0.001% of 0 based on the first 6 scans
# import numpy as np
# unique_values, value_counts = np.unique(dataset.labels.numpy(), return_counts=True)
# # Print the summary
# for value, count in zip(unique_values, value_counts):
#     print(f"{value}: {count} occurrences")

# implementing k-fold cross validation
NUM_FOLDS = 2

# Initialize a stratified k-fold splitter
kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

progress_bar = tqdm(total=NUM_FOLDS)

# Perform k-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset, labels)):
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    BATCH_SIZE = 5
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
        logging.info(f"Fold {fold + 1} - Test accuracy: {accuracy}")

    progress_bar.set_postfix({"Fold": fold + 1})
    progress_bar.update(1)
