import torch
import einops

from sklearn.cluster import KMeans


class Fuzzifier:
    def __init__(self, n_clusters, n_features, X_train):
        self.n_clusters = n_clusters
        self.n_features = n_features
        # Initialize cluster centers using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_train)
        self.cluster_centers = torch.tensor(kmeans.cluster_centers_)

        # Compute standard deviation for each cluster  
        labels = kmeans.labels_
        cluster_std_devs = []
        for i in range(n_clusters):
            cluster_points = X_train[labels == i]  # Select points in cluster i
            std_dev = torch.std(cluster_points, dim=0)  # Compute std deviation per feature
            cluster_std_devs.append(std_dev)
            
        self.std_devs = torch.stack(cluster_std_devs)

    def fuzzify(self, X, eps=1e-8):
        """
        Feature-wise Gaussian fuzzification
        """

        # Align dimensions
        X = X[:, :, None]                         # [B, F, 1]
        centers = self.cluster_centers.T[None, :, :]  # [1, F, C]
        stds = torch.clamp(self.std_devs.T[None, :, :], min=eps)

        # 1-D Gaussian per feature
        u = torch.exp(-((X - centers) ** 2) / (2 * stds ** 2))  # [B, F, C]

        # Flatten feature × cluster
        u = einops.rearrange(u, 'b f c -> b (f c)')

        # Normalize per sample (optional but consistent)
        return u / (u.sum(dim=1, keepdim=True) + eps)
    
    
class NormalizedFuzzifier:
    def __init__(self, n_clusters, n_features, X_train):
        assert n_features == 1, "NormalizedFuzzifier only supports a single feature for now"
        self.n_clusters = n_clusters
        self.X_train = X_train  # Store training data

        # Calculate the range for the feature and partition it into 'n_clusters'
        input_range = (X_train.min(), X_train.max())
        partition_width = (input_range[1] - input_range[0]) / (n_clusters - 1)

        # Define cluster centers and standard deviations (for simplicity, using equal widths)
        self.cluster_centers = torch.tensor([input_range[0] + i * partition_width for i in range(n_clusters)])
        self.std_devs = torch.tensor([partition_width / 3 for _ in range(n_clusters)])

    def fuzzify(self, X):
        """
        Fuzzifies the input data X using grid partitioning for a single feature.
        Each feature's range is divided into 'n_clusters' partitions with a Gaussian membership function.
        """
        u = torch.zeros((X.shape[0], self.n_clusters))

        # Calculate membership values for each partition (using Gaussian membership functions)
        for i in range(self.n_clusters):
            u[:, i] = torch.exp(-0.5 * ((X[:, 0] - self.cluster_centers[i]) / self.std_devs[i]) ** 2)
        return u
    
def fuzzify(train_data, val_data, test_data, params, logger):
    n_clusters = params['n_clusters']
    logger.info(f"Fuzzifying with {n_clusters} clusters")
    
    x_train = torch.stack([t.x for t in train_data])
    x_val = torch.stack([t.x for t in val_data])
    x_test = torch.stack([t.x for t in test_data])
    
    node_features = torch.cat([x_train, x_val, x_test], dim=0)
    
    num_nodes = x_train.shape[1]
    n_features = x_train.shape[2]
    # Concat nodes
    node_features = einops.rearrange(node_features, 'b n f -> (b n) f', n=num_nodes)
    x_train = einops.rearrange(x_train, 'b n f -> (b n) f', n=num_nodes)
    
    logger.info(f"x_train: {x_train}")
    
    fuzzifier_cls = NormalizedFuzzifier if params.get('fuzzifier', None) == 'normalized' else Fuzzifier
    logger.info(f"Using fuzzifier: {fuzzifier_cls}")
    fuzzifier = fuzzifier_cls(n_clusters, n_features, x_train)
    logger.info(f"Node Cluster centers: {fuzzifier.cluster_centers}")
    logger.info(f"Node Std Devs       : {fuzzifier.std_devs}")
    
    node_features = fuzzifier.fuzzify(node_features)
    # Reshape back to [time, nodes, features]
    node_features = einops.rearrange(node_features, '(b n) f -> b n f', n=num_nodes).float()
    
    if "edge_attr" in train_data[0]:
        num_edges = train_data[0].edge_index.shape[1]
        x_train_edges = torch.stack([t.edge_attr for t in train_data])
        x_train_edges = einops.rearrange(x_train_edges, 'b n f -> (b n) f', n=num_edges)
        
        edge_features = torch.cat([x_train_edges,
                                   torch.stack([t.edge_attr for t in val_data]),
                                   torch.stack([t.edge_attr for t in test_data])], dim=0)
        edge_features = einops.rearrange(edge_features, 'b n f -> (b n) f', n=num_edges)
        
        n_features = edge_features.shape[1]

        fuzzifier = Fuzzifier(n_clusters, n_features, x_train_edges)
        logger.info(f"Edge Cluster centers: {fuzzifier.cluster_centers}")
        logger.info(f"Edge Std Devs       : {fuzzifier.std_devs}")
        edge_features = fuzzifier.fuzzify(edge_features)
        # Reshape back to [time, edges, features]
        edge_features = einops.rearrange(edge_features, '(b n) f -> b n f', n=num_edges)
    else:
        edge_features = None
        
    print("Fuzzification completed, updating data")
    for i in range(len(train_data)):
        train_data[i].x = node_features[i]
        if "edge_attr" in train_data[i]:
            train_data[i].edge_attr = edge_features[i]
    for i in range(len(val_data)):
        val_data[i].x = node_features[len(train_data) + i]
        if "edge_attr" in val_data[i]:
            val_data[i].edge_attr = edge_features[len(train_data) + i]
    for i in range(len(test_data)):
        test_data[i].x = node_features[len(train_data) + len(val_data) + i]
        if "edge_attr" in test_data[i]:
            test_data[i].edge_attr = edge_features[len(train_data) + len(val_data) + i]
            
    print("Fuzzification step completed, now we have {} features".format(train_data[0].x.shape[1]))
        
    return train_data, val_data, test_data
