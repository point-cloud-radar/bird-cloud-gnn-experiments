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
from bird_cloud_gnn.early_stopper import EarlyStopper
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DATA_PATH = "data/full"
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
    num_neighbours=200,
    max_poi_per_label=5000,
    max_edge_distance=5_000.0,
)


num_examples = len(dataset)
num_train = int(num_examples * 0.8)

rng = torch.Generator().manual_seed(42)

train_idx, test_idx = torch.utils.data.random_split(
    range(num_examples),
    [num_train, num_examples - num_train],
    generator=rng,
)

# train_idx = torch.arange(num_train)
# test_idx = torch.arange(num_train, num_examples)

model = GCN(len(dataset.features), 32, 2)

train_dataloader, test_dataloader = get_dataloaders(
    dataset, train_idx, test_idx, batch_size=512
)

num_epochs = 4

batch_size = 512
writer = SummaryWriter()
early_stopper = EarlyStopper(patience=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
progress_bar = tqdm(total=num_epochs)

for epoch in range(num_epochs):
    train_loss = 0.0
    num_correct = 0
    num = 0
    model.train()
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata["x"].float())
        loss = torch.nn.functional.cross_entropy(pred, labels)
        train_loss += loss
        num_correct += (pred.argmax(1) == labels).sum().item()
        num += len(labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", num_correct / num, epoch)

    test_loss = 0.0
    num_correct = 0
    num = 0
    model.eval()
    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata["x"].float())
        test_loss += torch.nn.functional.cross_entropy(pred, labels)
        num_correct += (pred.argmax(1) == labels).sum().item()
        num += len(labels)

    # print(f"{epoch}: Loss/test {test_loss}")
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", num_correct / num, epoch)

    progress_bar.set_postfix({"Epoch": epoch})
    progress_bar.update(1)

    if early_stopper.early_stop(float(test_loss)):
        break
