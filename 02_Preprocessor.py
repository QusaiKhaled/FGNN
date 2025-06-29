import torch
import numpy as np
from torch_geometric.data import Data

# --- MODIFIED: Preprocessing for Semi-Supervised Learning ---
class SemiSupervisedPreprocessor:
    """
    This preprocessor splits time-series graph data for semi-supervised learning.
    It divides sequences into training and testing by chronological order,
    imputes missing values, normalizes features, and constructs fixed-size windows.
    """
    def __init__(self, data, window_size=288, stride=144, train_ratio=0.7, max_windows=None):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.train_ratio = train_ratio
        self.max_windows = max_windows
        self.num_nodes = data.x.shape[0]
        self.num_node_features = 1
        self.num_timesteps = data.x.shape[1] // self.num_node_features

    def _split_time(self):
        split_idx = int(self.num_timesteps * self.train_ratio)
        seq = self.data.x.view(self.num_nodes, self.num_timesteps, self.num_node_features)
        return (seq[:, :split_idx, :], self.data.leak_target[:, :split_idx]), \
               (seq[:, split_idx:, :], self.data.leak_target[:, split_idx:])

    def _impute_and_normalize(self, x_train, x_test):
        # Convert to NumPy for efficient nan operations
        x_tr = x_train.cpu().numpy()
        x_te = x_test.cpu().numpy()

        # Impute missing values using training set mean
        mean_val = np.nanmean(x_tr)
        mean_val = 0.0 if np.isnan(mean_val) else mean_val
        x_tr[np.isnan(x_tr)] = mean_val
        x_te[np.isnan(x_te)] = mean_val

        # Back to tensor
        x_tr = torch.from_numpy(x_tr).float()
        x_te = torch.from_numpy(x_te).float()

        # Normalize by training statistics
        mu = x_tr.mean(dim=(0, 1), keepdim=True)
        sigma = x_tr.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
        return (x_tr - mu) / sigma, (x_te - mu) / sigma

    def _create_windows(self, x, y, is_training_set):
        train_windows, test_windows = [], []
        num_steps = x.shape[1]
        num_to_process = (num_steps - self.window_size) // self.stride + 1
        if self.max_windows:
            num_to_process = min(num_to_process, self.max_windows)

        for i in range(num_to_process):
            start = i * self.stride
            end = start + self.window_size
            x_w = x[:, start:end, :].reshape(self.num_nodes, -1)
            y_w = y[:, start:end]

            # Build edge-level targets
            edge_leak = torch.zeros(self.data.edge_index.shape[1], dtype=torch.float)
            for j, (u, v) in enumerate(self.data.edge_index.t()):
                edge_leak[j] = float((y_w[u].max() > 0) or (y_w[v].max() > 0))

            graph = Data(x=x_w, edge_index=self.data.edge_index, edge_leak_target=edge_leak)

            if is_training_set:
                if edge_leak.sum() == 0:
                    train_windows.append(graph)
            else:
                test_windows.append(graph)

        return train_windows, test_windows

    def preprocess(self):
        # Split chronologically
        (xt, yt), (xv, yv) = self._split_time()
        # Impute & normalize
        xt_norm, xv_norm = self._impute_and_normalize(xt, xv)
        # Create windows
        train_data, _ = self._create_windows(xt_norm, yt, is_training_set=True)
        _, test_data = self._create_windows(xv_norm, yv, is_training_set=False)
        return train_data, test_data
