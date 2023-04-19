import os
import time
import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from bird_cloud_gnn.gnn_model import GCN
from bird_cloud_gnn.radar_dataset import RadarDataset


# TODO adjust methods to allow reading .gz
data_path = "../data/manual_annotations/"
print(f"Files in the directory: {os.listdir(data_path)}")

# TODO include other relevant features
features = [
    "range",
    "azimuth",
    "elevation",
    "x",
    "y",
    "z",
    "DBZH",
    "DBZV",
]
# 5 first files result in 5981 graphs
st = time.process_time()

dataset = RadarDataset(
    data_folder=data_path,
    features=features,
    target="BIOLOGY",
    max_distance=500.0,
    min_neighbours=100,
    max_edge_distance=50.0,
)

cpu_time = time.process_time() - st
print("Execution time:", time.strftime("%H:%M:%S", time.gmtime(cpu_time)))

for graph, label in dataset:
    print(graph, label)

num_graphs = len(dataset)
num_train = int(num_graphs * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_graphs))

train_dataloader = GraphDataLoader(
    dataset,
    sampler=train_sampler,
    batch_size=5,
    drop_last=False,
)
test_dataloader = GraphDataLoader(
    dataset,
    sampler=test_sampler,
    batch_size=5,
    drop_last=False,
)

model = GCN(len(dataset.features), 16, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(20):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata["x"].float())
        loss = torch.nn.functional.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert pred.dim() == model.num_classes

num_correct = 0
num_tests = 0

for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata["x"].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)
    assert pred.dim() == model.num_classes

# TODO Check the following
# pkl and bin in the same working dir intended behaviour?
# 2 pkl and 2 bin for 3 files?
# 3 files 452 graphs whereas 5 files 5981 graphs?
