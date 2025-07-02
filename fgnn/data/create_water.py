import pandas as pd
import networkx as nx
import numpy as np
import random
import torch

from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm


# === Helper: Read Excel with proper decimal handling ===
def read_sheet_with_decimal(file, sheet, index_col):
    """
    Reads an Excel sheet, converts comma decimals to dots, handles missing values,
    and converts index to datetime if possible.
    """
    df = pd.read_excel(
        file, sheet_name=sheet, dtype=str
    )  # Read as string to handle custom decimal format
    df = df.set_index(index_col)  # Set the timestamp or identifier as index

    # Replace ',' with '.', replace 'nan' string with np.nan, convert to float
    for col in df.columns:
        df[col] = (
            df[col]
            .str.replace(",", ".", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )

    # Try converting index to datetime
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df


def make_undirected(data):
    """
    Converts a directed PyTorch Geometric graph to an undirected one by:
    - Duplicating each edge in reverse direction
    - Duplicating associated edge attributes (e.g., leak targets, pipe names)
    """

    # If edge input features exist (not used in our case), duplicate them for reversed edges
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)

    # Reverse the direction of all existing edges: (u, v) -> (v, u)
    reversed_edges = data.edge_index[[1, 0], :]

    # Concatenate original and reversed edges to make the graph undirected
    data.edge_index = torch.cat([data.edge_index, reversed_edges], dim=1)

    # If edge target features (leak targets) are available per edge, duplicate them for the reversed edges
    if hasattr(data, "leak_target") and data.leak_target is not None:
        data.leak_target = torch.cat([data.leak_target, data.leak_target], dim=0)

    # Duplicate pipe names for reversed edges (if available)
    if hasattr(data, "pipe_names"):
        data.pipe_names = data.pipe_names + data.pipe_names

    return data

def create_graph_water(root="Data"):
    """
    Create a graph dataset for water leak detection from an Excel file.

    This function reads
    Returns:
        pd.DataFrame: Processed DataFrame with stripped whitespace.
    """

    root = Path(root)
    excel_file = root / "GraphExcel.xlsx"
    # Read the Excel file named 'GraphExcel.xlsx' in the same directory
    df = pd.read_excel(excel_file)

    # Strip whitespace from all string entries in the DataFrame
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # using networkx library to construct an undirected graph from the excel file

    # initialize the undirected graph
    G = nx.Graph()

    # Add edges to the graph from the Dataframe
    # Each pipe connects Node 1 and Node 2
    for _, row in df.iterrows():
        G.add_edge(row["Node1"], row["Node2"], pipe=row["Pipe"])

    # check all nodes and edges are mapped correctly
    print("Random node check:")
    # Print the full name of one random node
    random_node = random.choice(list(G.nodes()))
    print("Random node:", random_node)

    # Print the full name of one random edge (with data)
    random_edge = random.choice(list(G.edges(data=True)))
    print("Random edge:", random_edge)

    ### `Step 2: Add your data to graph G to make Water_Graph.pt`
    # === Load pressure data only ===
    pressure_df = read_sheet_with_decimal(
        root / "2018_SCADA.xlsx", "Pressures (m)", "Timestamp"
    )
    # This will be used as node features (pressure over time for each node)

    # === Graph setup ===
    G_nodes = list(G.nodes())  # List of nodes in the graph
    num_nodes = len(G_nodes)  # Total number of nodes
    num_timesteps = len(pressure_df)  # Number of time steps in pressure data

    # === Node Features (only pressure) ===
    node_features = []  # Will hold pressure time series per node
    feature_mask = []  # Will mark which nodes have actual pressure data

    bar = tqdm(
        G_nodes, desc="Processing nodes", unit="node", leave=False
    )  # Progress bar for nodes

    for node in bar:
        if node in pressure_df.columns:
            # If pressure data is available for this node
            p = pressure_df[node].values  # Time series of pressures
            p_mask = 1  # Data is available
        else:
            # If pressure data is missing, fill with NaNs
            p = np.full(num_timesteps, np.nan)
            p_mask = 0  # Data not available

        node_feat = np.expand_dims(
            p, axis=1
        )  # Convert shape [T] to [T, 1] for consistency
        node_features.append(node_feat)  # Add to feature list
        feature_mask.append([p_mask])  # Append data availability mask

    # Stack node features to form a 3D array: [num_nodes, timesteps, 1]
    node_features = np.stack(node_features)

    # Stack feature mask to form a 2D array: [num_nodes, 1]
    feature_mask = np.array(feature_mask)

    # Flatten time dimension to create 2D tensor for GNN: [num_nodes, timesteps]
    x = torch.from_numpy(node_features).float().view(num_nodes, -1)

    # === Edge Index ===
    # Convert edges into PyG format: [2, num_edges] with node indices
    edge_index = []
    for u, v in G.edges():
        edge_index.append(
            [G_nodes.index(u), G_nodes.index(v)]
        )  # Find index of each node

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # === Build graph ===
    data = Data()
    data.x = x  # Node features (flattened pressure series)
    data.edge_index = edge_index  # Connectivity information
    data.feature_mask = torch.tensor(
        feature_mask, dtype=torch.float
    )  # Mask for missing features

    # === Load leakage target CSV ===
    # This file contains leakage values per pipe over time (target variable)
    leak_df = pd.read_csv(
        "Data/2018_Leakages.csv",
        sep=";",  # Semicolon-separated
        decimal=",",  # Comma as decimal separator
        index_col="Timestamp",
        parse_dates=True,  # Parse timestamp into datetime
        low_memory=False,
    )

    num_edges = edge_index.size(1)  # Number of edges (pipes)
    leak_target = np.full((num_edges, len(leak_df)), np.nan)  # Placeholder for targets
    edge_pipe_names = []  # To store pipe names for each edge

    # === Match leak data to edges ===
    # For each edge, try to find the corresponding pipe name and extract its leakage data
    for i, (u, v, attr) in enumerate(G.edges(data=True)):
        pipe_name = attr.get("pipe")  # Get 'pipe' attribute of edge
        edge_pipe_names.append(pipe_name)  # Store pipe name for reference

        if pipe_name in leak_df.columns:
            # If leakage data is available for this pipe, copy it into target array
            leak_target[i, :] = leak_df[pipe_name].values

    # Add leakage target as edge features
    data.leak_target = torch.tensor(leak_target, dtype=torch.float)
    data.pipe_names = edge_pipe_names  # Save pipe names for possible future reference

    print("✅ Graph with pressure-only node features and edge leak targets created")

    # Basic graph info
    num_nodes = data.x.shape[0]
    num_features = data.x.shape[1]
    num_edges = data.edge_index.shape[1]

    # Updated: Only pressure data remains
    feature_names = ["pressure"]
    feature_dim_per_timestep = len(feature_names)
    time_series_length = num_features // feature_dim_per_timestep

    print("Basic Graph Information:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Time series length (number of timesteps): {time_series_length}")
    print(f"Features per node: {feature_names}")

    # Use feature_mask to count how many nodes have pressure data
    if hasattr(data, "feature_mask"):
        available_nodes = int(data.feature_mask.sum().item())
        print(
            f"Number of nodes with available pressure data: {available_nodes} / {num_nodes}"
        )

    # If leak targets exist, analyze leaky edges
    if hasattr(data, "leak_target"):
        # An edge is considered leaky if it has at least one non-NaN value
        leak_target = data.leak_target
        leaky_edges_mask = ~torch.isnan(leak_target).all(dim=1)
        num_leaky_edges = leaky_edges_mask.sum().item()
        print(
            f"Number of leaky edges (with leak data): {num_leaky_edges} / {num_edges}"
        )

    # Optionally, print a few pipe names as examples if available
    if hasattr(data, "pipe_names"):
        print(f"Example pipe names: {data.pipe_names[:5]}")

    ### `Step 3: Make the graph undirected`
    # === Convert the graph to undirected by adding reversed edges and duplicating edge-level data ===
    data = make_undirected(data)

    # Count total number of edges
    num_edges = data.edge_index.shape[1]

    # Count number of edges with leakage data (should match leak_target rows)
    num_leak_edges = data.leak_target.shape[0] if hasattr(data, "leak_target") else 0

    print(f"Total number of edges: {num_edges}")
    print(
        "✅ Number of edges is now doubled due to the undirected nature of the graph."
    )

    ### `Step 4: Binarize edges as explained below`
    # **our edge classification task means that an edge is 1 if its leaky and 0 if its none-leaky, but the existing data only have timeseries for the edges that are leaky, while all other edges were set to NaN. We do know that all edges without timeseries data are not leaky so we set their values to 0 instead of NaN, next thing, since we dont care about the amount of leak in edges but only want to determine which edges are leaky we change the type of data in the leaky edges from continuous to binary so for all leaky edges we set their values to 1 and for all non-leaky edges we set their values to 0**

    # Ensure leak_target exists
    if not hasattr(data, 'leak_target') or data.leak_target is None:
        raise ValueError("The graph does not contain 'leak_target' data.")

    # Tensor shape: [num_edges, timesteps]
    leak_target = data.leak_target

    # Check which rows have any non-NaN values
    has_leak_data = ~torch.isnan(leak_target).all(dim=1)

    # Count
    num_edges_with_target = has_leak_data.sum().item()
    num_edges_without_target = leak_target.size(0) - num_edges_with_target

    # Print
    print("📌 Leak Target Availability")
    print("----------------------------")
    print(f"Total edges           : {leak_target.size(0)}")
    print(f"Edges with targets    : {num_edges_with_target}")
    print(f"Edges without targets : {num_edges_without_target}")
    
    # setting target edges without leak to zero leak value

    # Check that leak_target exists
    if not hasattr(data, 'leak_target') or data.leak_target is None:
        raise ValueError("This graph does not contain 'leak_target' data.")

    # Identify edges where all values are NaN
    leak_target = data.leak_target  # shape: [num_edges, timesteps]
    mask_all_nan = torch.isnan(leak_target).all(dim=1)  # shape: [num_edges]

    # Set all-NaN rows to 0
    data.leak_target[mask_all_nan] = 0.0

    print(f"✅ Set leak target to 0 for {mask_all_nan.sum().item()} edges with no leak data.")
    
    
    # Check that leak_target exists
    if not hasattr(data, 'leak_target') or data.leak_target is None:
        raise ValueError("This graph does not contain 'leak_target' data.")

    # Binarize: set all values > 0 to 1, keep 0 as is
    data.leak_target = (data.leak_target > 0).float()
    
    print("✅ Leak target has been binarized for leak localization (0 = no leak, 1 = leak).")

    # Save the updated graph
    torch.save(data, root / 'Water_Graph.pt')
    print(f"✅ Graph saved as '{root / "Water_Graph.pt"}'.")