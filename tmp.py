import time

from bird_cloud_gnn.radar_dataset import RadarDataset

st = time.perf_counter()
try:
    dataset = RadarDataset(
        "data/chunk1",
        ["x", "y", "z"],
        "BIOLOGY",
        min_neighbours=25,
        max_distance=500.0,
        max_edge_distance=250.1,
    )
    print(dataset)
    print(dataset[0][0])
except ValueError as ex:
    print("Error:", ex)
print("Total time:", time.perf_counter() - st)
