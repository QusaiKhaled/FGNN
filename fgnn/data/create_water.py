import pandas as pd
import networkx as nx
import numpy as np
import random
import torch
import einops

from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm


# === Helper: Read Excel with proper decimal handling ===
def read_sheet_with_decimal(file, sheet, index_col):
    """
    Reads an Excel sheet, converts comma decimals to dots, handles missing values,
    and converts index to datetime if possible.
    """
    file = Path(file)
    # if xlsx
    if file.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(
            file, sheet_name=sheet, dtype=str
        )  # Read as string to handle custom decimal format
    elif file.suffix == ".csv":
        df = pd.read_csv(
            file, sep=";", dtype=str
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

def create_propagated_features(G, pressure_df, num_timesteps, static_df=None):
    
    num_nodes = len(G.nodes)  # Total number of nodes

    valid_pressure_nodes = [n for n in G.nodes if n in pressure_df.columns]
    node_pressure_data = {
        n: np.expand_dims(pressure_df[n].values, axis=1)
        for n in valid_pressure_nodes
    }

    node_features = []
    static_features = []
    feature_mask = []  # 1 if original data, 0 if imputed or missing

    bar = tqdm(G.nodes, desc="Propagating features", unit="node", leave=False)

    for node in bar:
        if node in pressure_df.columns:
            pressure = node_pressure_data[node]
            dist = 0
            mask = [1]
        else:
            try:
                distances = nx.single_source_shortest_path_length(G, node)
                candidates = [(src, d) for src, d in distances.items() if src in node_pressure_data]
                if not candidates:
                    raise ValueError("No reachable pressure-known node")

                nearest_node, dist = min(candidates, key=lambda x: x[1])
                pressure = node_pressure_data[nearest_node]
                mask = [0]

            except (nx.NetworkXNoPath, ValueError):
                # Warning: No path to any pressure node
                bar.write(f"Warning: No reachable pressure node for {node}. Imputing with NaN.")
                # Impute with NaN for this node
                pressure = np.full((num_timesteps, 1), np.nan)
                dist = np.inf
                mask = [0]
                
        # Add static features if available
        if static_df is not None and node in static_df.index:
            static_feat = static_df.loc[node].values.tolist()
            static_feat = [dist] + static_feat
        else:
            static_feat = [-1] * (static_df.shape[1] if static_df is not None else 0)
            static_feat = [dist] + static_feat

        node_features.append(pressure)
        static_features.append(static_feat)
        feature_mask.append(mask)


    # Stack node features to form a 3D array: [num_nodes, timesteps, 1]
    node_features = np.stack(node_features)

    # Stack feature mask to form a 2D array: [num_nodes, 1]
    # feature_mask = np.array(feature_mask)
    static_features = torch.from_numpy(np.array(static_features))
    node_features = torch.from_numpy(node_features).float() # [num_nodes, timesteps, features]

    return node_features, static_features


def create_masked_features(G_nodes, pressure_df, num_timesteps, static_df=None):
    # === Node Features (only pressure) ===
    node_features = []  # Will hold pressure time series per node
    feature_mask = []  # Will mark which nodes have actual pressure data
    num_nodes = len(G_nodes)

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
        
    return x, feature_mask

def create_graph_water(parameters):
    """
    Create a graph dataset for water leak detection from an Excel file.
    Returns a PyG Data object (saved to disk and also returned).
    """
    root = parameters.get("root", "Data")  # Default root directory
    feature_distance = parameters.get("feature_distance", False)
    
    root = Path(root)
    excel_file = root / "GraphExcel.xlsx"
    # Read the Excel file named 'GraphExcel.xlsx' in the same directory
    df = pd.read_excel(excel_file)

    # Strip whitespace from all string entries in the DataFrame
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # initialize the undirected graph
    G = nx.Graph()

    # Add edges to the graph from the Dataframe
    for _, row in df.iterrows():
        G.add_edge(row["Node1"], row["Node2"], pipe=row["Pipe"])

    print("Random node check:")
    random_node = random.choice(list(G.nodes()))
    print("Random node:", random_node)
    random_edge = random.choice(list(G.edges(data=True)))
    print("Random edge:", random_edge)

    years = parameters.get("years", [2018])
    pressure_suffix = parameters.get("pressure_suffix", "_SCADA.xlsx")
    
    pressure_dfs = []
    year_len = []
    for year in years:
        pressure_file = root / f"{year}{pressure_suffix}"
        pressure_df_year = read_sheet_with_decimal(
            pressure_file, "Pressures (m)", "Timestamp"
        )
        pressure_dfs.append(pressure_df_year)
        year_len.append(len(pressure_df_year))
    # Concatenate along the time axis (index)
    pressure_df = pd.concat(pressure_dfs, axis=0)

    # --- Ensure pressure_df index is datetime (important for timestamps) ---
    pressure_df.index = pd.to_datetime(pressure_df.index)

    # === Graph setup ===
    G_nodes = list(G.nodes())  # List of nodes in the graph (preserves order)
    num_nodes = len(G_nodes)  # Total number of nodes
    num_timesteps = len(pressure_df)  # Number of time steps in pressure data
    
    static_features_list = parameters.get("static", [])
    static_features_path = root / "static2.xlsx"
    if static_features_list:
        static_df = pd.read_excel(static_features_path).loc[:, ["Node"] + static_features_list]
        static_df = static_df.set_index('Node')
    else:
        static_df = None

    if feature_distance:
        x, static_features = create_propagated_features(
            G, pressure_df, num_timesteps, static_df
        )
    else:
        x, static_features = create_masked_features(
            G_nodes, pressure_df, num_timesteps, static_df
        )  # Create node features and mask
        

    # === Edge Index ===
    edge_index = []
    for u, v in G.edges():
        edge_index.append([G_nodes.index(u), G_nodes.index(v)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # === Build graph ===
    data = Data()
    data.x = x  # Node features (flattened pressure series)
    data.edge_index = edge_index  # Connectivity information
    data.static_features = torch.tensor(static_features, dtype=torch.float) if static_features is not None else None
    
    # Add the year separation info to the data object
    data.year_len = torch.tensor(year_len, dtype=torch.long)

    # --- NEW: Save node names (strings) so you can map node indices back to original names ---
    data.node_names = G_nodes  # list of node name strings

    # --- NEW: Attach timestamps corresponding to the timesteps in pressure_df ---
    # 1) keep a pandas DatetimeIndex for convenience/inspection
    data.timestamp_index = pressure_df.index  # pandas.DatetimeIndex (useful interactively)
    # 2) also store as an int64 tensor (nanoseconds since epoch) — easy to use in ML pipelines
    timestamps_ns = pressure_df.index.view('int64').astype('int64')  # pure np.ndarray
    data.timestamps = torch.from_numpy(timestamps_ns).long()

    # === Load leakage target CSVs for all specified years and concatenate ===
    leak_dfs = []
    for year in years:
        leak_file = root / f"{year}_Leakages.csv"
        leak_df_year = pd.read_csv(
            leak_file,
            sep=";",  # Semicolon-separated
            decimal=",",  # Comma as decimal separator
            index_col="Timestamp",
            parse_dates=True,  # Parse timestamp into datetime
            low_memory=False,
        )
        leak_dfs.append(leak_df_year)
    leak_df = pd.concat(leak_dfs, axis=0)

    num_edges = edge_index.size(1)  # Number of edges (pipes)
    leak_target = np.full((num_edges, len(leak_df)), np.nan)  # Placeholder for targets
    edge_pipe_names = []

    for i, (u, v, attr) in enumerate(G.edges(data=True)):
        pipe_name = attr.get("pipe")
        edge_pipe_names.append(pipe_name)
        if pipe_name in leak_df.columns:
            leak_target[i, :] = leak_df[pipe_name].values

    data.leak_target = torch.tensor(leak_target, dtype=torch.float)
    data.pipe_names = edge_pipe_names  # Save pipe names for possible future reference

    print("✅ Graph with pressure-only node features and edge leak targets created")

    # Basic graph info
    num_nodes = data.x.shape[0]
    num_features = data.x.shape[1]
    num_edges = data.edge_index.shape[1]

    feature_names = ["pressure"]
    feature_dim_per_timestep = len(feature_names)
    time_series_length = num_features // feature_dim_per_timestep

    print("Basic Graph Information:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Time series length (number of timesteps): {time_series_length}")
    print(f"Features per node: {feature_names}")

    if hasattr(data, "feature_mask"):
        available_nodes = int(data.feature_mask.sum().item())
        print(
            f"Number of nodes with available pressure data: {available_nodes} / {num_nodes}"
        )

    if hasattr(data, "leak_target"):
        leak_target = data.leak_target
        leaky_edges_mask = ~torch.isnan(leak_target).all(dim=1)
        num_leaky_edges = leaky_edges_mask.sum().item()
        print(
            f"Number of leaky edges (with leak data): {num_leaky_edges} / {num_edges}"
        )

    if hasattr(data, "pipe_names"):
        print(f"Example pipe names: {data.pipe_names[:5]}")

    # Make undirected (duplicates edges/edge-level data)
    data = make_undirected(data)

    num_edges = data.edge_index.shape[1]
    num_leak_edges = data.leak_target.shape[0] if hasattr(data, "leak_target") else 0

    print(f"Total number of edges: {num_edges}")
    print("✅ Number of edges is now doubled due to the undirected nature of the graph.")

    # --- Binarize edges as before ---
    if not hasattr(data, 'leak_target') or data.leak_target is None:
        raise ValueError("The graph does not contain 'leak_target' data.")
    leak_target = data.leak_target
    has_leak_data = ~torch.isnan(leak_target).all(dim=1)
    num_edges_with_target = has_leak_data.sum().item()
    num_edges_without_target = leak_target.size(0) - num_edges_with_target

    print("📌 Leak Target Availability")
    print("----------------------------")
    print(f"Total edges           : {leak_target.size(0)}")
    print(f"Edges with targets    : {num_edges_with_target}")
    print(f"Edges without targets : {num_edges_without_target}")
    
    mask_all_nan = torch.isnan(leak_target).all(dim=1)
    data.leak_target[mask_all_nan] = 0.0
    print(f"✅ Set leak target to 0 for {mask_all_nan.sum().item()} edges with no leak data.")
    
    data.leak_target = (data.leak_target > 0).float()
    print("✅ Leak target has been binarized for leak localization (0 = no leak, 1 = leak).")

    # Save the updated graph
    filename = parameters.get("filename", "Water_Graph.pt")
    torch.save(data, root / filename)
    print(f"✅ Graph saved as '{root / filename}'.")

    # Return the Data object too (convenient for immediate use)
    return data
