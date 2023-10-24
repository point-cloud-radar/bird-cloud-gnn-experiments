import os
import torch
import configargparse

parser = configargparse.ArgParser(
    description="Train bird cloud model",
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
)
parser.add("-v", "--verbose", action="store_true", help="Increase verbosity")
parser.add(
    "-c", "--config", required=False, is_config_file=True, help="Config file path"
)
parser.add(
    "-s",
    "--seed",
    default=42,
    env_var="SEED",
    type=int,
    help="The seed for calculations",
)
parser.add(
    "--num-nodes", default=50, type=int, help="The number of nodes for each graph created"
)
parser.add("--num-epochs", default=500, type=int, help="The number epochs to run")
parser.add("--learning-rate", default=0.005, type=float, help="The learning rate")
parser.add(
    "--max-edge-distance",
    default=650.0,
    type=float,
    help="The maximal distance over which nodes in the graph are connected",
)
parser.add(
    "--batch-size", default=512, type=int, help="The batch size for data loadin durring model training"
)
parser.add(
    "--max-poi-per-label", default=5000, type=int, help="The number of poi's per label"
)
parser.add(
    "--num-hidden-features", default=16, type=int, help="The number of hidden features"
)
parser.add(
    "--skip-missing-indicator-columns",
    "--smic",
    action="store_false",
    help="Do not generate additional missing data indicator columns",
)
parser.add(
    "--remove-edges-to-poi",
    "--retp",
    action="store_false",
    help="Omit the the default edges to the point of interest",
)
parser.add(
    "--data-path",
    default="/opt/bird-data/",
    env_var="DATA_PATH",
    help="The path to the data directory",
)
parser.add('--sch-multisteplr-gamma', default=1, type=float, 
           help="The exponential decay of the learning rate, when using a stepped scheduling approach")
parser.add('--sch-multisteplr-milestones', nargs="+", 
           type=int,default=100, help="The epochs at which milestone decay should be executed")

parser.add('--sch-explr-gamma', default=1, type=float, help="The exponential decay of the learning rate per epoch")
args = parser.parse_args()
config = vars(args)
if config["verbose"]:
    parser.print_values()

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
    data=config["data_path"],
    features=features,
    target="BIOLOGY",
    num_nodes=NUM_NODES,
    max_poi_per_label=5000,  # 500
    max_edge_distance=MAX_EDGE_DISTANCE,
    use_missing_indicator_columns=True,
    add_edges_to_poi=True,
)
print(f"Dataset size: {len(dataset)}")

if isinstance(config['sch_multisteplr_milestones'], int):
    config['sch_multisteplr_milestones']=[config['sch_multisteplr_milestones']]
num_examples = len(dataset)
num_train = int(num_examples * 0.8)
np.random.seed(config["seed"])
train_idx = np.random.choice(num_examples, num_train, replace=False)
test_idx = np.setdiff1d(np.arange(0, num_examples), train_idx, assume_unique=True)

model = GCN(len(dataset.features), [(16, nn.ReLU()), (16, nn.ReLU()), (2, None)])

train_dataloader, test_dataloader = get_dataloaders(
    dataset, train_idx, test_idx, batch_size=config["batch_size"]
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
torch.save(model.state_dict(), pth+".pt")
