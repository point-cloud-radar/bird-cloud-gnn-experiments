import os
import torch
import configargparse
from torch import nn
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
parser.add('--sample-origin', '--so', action='store_true',help='If set the test set is created per origin')

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
import pandas as pd
import numpy as np

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
torch.manual_seed(config["seed"])

dataset = RadarDataset(
    data=config["data_path"],
    features=features,
    target="BIOLOGY",
    num_nodes=config["num_nodes"],
    max_poi_per_label=config["max_poi_per_label"],
    max_edge_distance=config["max_edge_distance"],
    use_missing_indicator_columns=config["skip_missing_indicator_columns"],
    add_edges_to_poi=config["remove_edges_to_poi"],
)

if config["verbose"]:
    print(f"Dataset size: {len(dataset)}")

if isinstance(config['sch_multisteplr_milestones'], int):
    config['sch_multisteplr_milestones']=[config['sch_multisteplr_milestones']]
num_examples = len(dataset)
num_train = int(num_examples * 0.8)
np.random.seed(config["seed"])
if config['sample_origin']:
    # sample per origin, by first tabulating the count per origin, randomizing the order, abs cumilatively suming this
    # Note that the test split will be minimal 20% but might be slightly larger as there is not always the right number of graphs per origin 
    train_origins=dataset.origin.value_counts().sample(frac=1).cumsum().loc[lambda x :x <= num_train].index
    train_idx=pd.Series(dataset.origin).index.where(dataset.origin.isin(train_origins)).dropna().to_numpy().astype(int)
    test_origins= dataset.origin.unique()[~dataset.origin.unique().isin(train_origins)]
    if config["verbose"]:
        print(f"# target train samples: {num_train} - resulting # train: {len(train_idx)} - out of: {num_examples}")
        print(f"Test origins (n={len(test_origins)}): " +" ".join(test_origins))
        print(f"Train origins (n={len(train_origins)}): " +" ".join(train_origins))
        print("Testing balance: "+ str(dataset.labels[pd.Series(dataset.origin).index.where(~dataset.origin.isin(train_origins)).dropna().to_numpy().astype(int)].numpy().mean()))
        print("Training balance: "+ str(dataset.labels[train_idx].numpy().mean()))
        print(f"Overall balance: {dataset.labels.numpy().mean()}")

else:
    train_idx = np.random.choice(num_examples, num_train, replace=False)
test_idx = np.setdiff1d(np.arange(0, num_examples), train_idx, assume_unique=True)

model = GCN(len(dataset.features), [(config["num_hidden_features"], nn.ReLU()),
                                    (config["num_hidden_features"], nn.ReLU()),
                                    (2,None)])

train_dataloader, test_dataloader = get_dataloaders(
    dataset, train_idx, test_idx, batch_size=config["batch_size"]
)
pth= "/".join(
                [
                    "runs",
                    dataset.oneline_description(),
                    model.oneline_description(),
                    f"LR_{config['learning_rate']}-BS_{config['batch_size']}-SEED_{config['seed']}-SMG_{config['sch_multisteplr_gamma']}-SEG_{config['sch_explr_gamma']}-OS_{config['sample_origin']}",
                ])
if(config['verbose']):
    print(f"Using path: {pth}")
callback = CombinedCallback(
    [
        TensorboardCallback(
            log_dir=pth
            ,
        ),
        EarlyStopperCallback(patience=500),
    ]
)

model.fit_and_evaluate(
    train_dataloader,
    test_dataloader,
    callback,
    learning_rate=config["learning_rate"],
    num_epochs=config["num_epochs"],
    sch_explr_gamma=config["sch_explr_gamma"],
    sch_multisteplr_gamma=config['sch_multisteplr_gamma'],
    sch_multisteplr_milestones=config['sch_multisteplr_milestones'],
)

torch.save(model.state_dict(), pth+".pt")
