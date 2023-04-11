import networkx as nx
import plotly.express as px
import plotly.graph_objects as go


def show_one_graph(graph, positions):
    """Show one graph in three-dimensions.

    This function shows one of the Graph objects inside RadarDataset using the
    positions matrix to plot it in three dimensions.

    Args:
        graph (DGLGraph):
            Graph object. To access the graph at index i of a RadarDataset
            variable called `dataset`, use `dataset[i][0]`.
        positions (ndarray):
            A (number of nodes)x3 matrix representing the cartesian coordinates
            of each node. Can be obtained, for instance, if you have a features
            DataFrame called `features`, by calling `features[['x','y','z']].values`.
    """
    G = graph.to_networkx()

    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = positions[edge[0], :]
        x1, y1, z1 = positions[edge[1], :]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        line=dict(width=2.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = positions[node, :]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append("# of connections: " + str(len(adjacencies[1])))

    # node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Graph Visualisation",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
        ),
    )
    fig.show()
