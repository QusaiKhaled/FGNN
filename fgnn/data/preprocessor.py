import torch
import numpy as np
from torch_geometric.data import Data

import einops

# --- MODIFIED: Preprocessing for Semi-Supervised Learning ---
class SemiSupervisedPreprocessor:
    """
    This preprocessor splits time-series graph data for semi-supervised learning.
    It divides sequences into training, validation, and testing by chronological order,
    imputes missing values, normalizes features, and constructs fixed-size windows.
    """
    def __init__(self, data, window_size=288, stride=144, train_ratio=0.6, val_ratio=0.2,
                 max_windows=None, num_nodes_features=1, anomaly_detection=False):
        assert train_ratio + val_ratio < 1.0, "Train + val ratio must be < 1.0 (remainder is test)"
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.max_windows = max_windows
        self.num_nodes = data.x.shape[0]
        self.num_node_features = num_nodes_features
        self.num_timesteps = data.x.shape[1] // self.num_node_features
        self.anomaly_detection = anomaly_detection

    def _split_time(self):
        train_end = int(self.num_timesteps * self.train_ratio)
        val_end = int(self.num_timesteps * (self.train_ratio + self.val_ratio))

        seq = self.data.x.view(self.num_nodes, self.num_timesteps, self.num_node_features)

        return (seq[:, :train_end, :], self.data.leak_target[:, :train_end]), \
               (seq[:, train_end:val_end, :], self.data.leak_target[:, train_end:val_end]), \
               (seq[:, val_end:, :], self.data.leak_target[:, val_end:])

    def _impute_and_normalize(self, x_train, x_val, x_test):
        # Convert to NumPy for efficient nan operations
        x_tr = x_train.cpu().numpy()
        x_va = x_val.cpu().numpy()
        x_te = x_test.cpu().numpy()

        # Impute missing values using training set mean
        mean_val = np.nanmean(x_tr)
        mean_val = 0.0 if np.isnan(mean_val) else mean_val
        x_tr[np.isnan(x_tr)] = mean_val
        x_va[np.isnan(x_va)] = mean_val
        x_te[np.isnan(x_te)] = mean_val

        # Back to tensor
        x_tr = torch.from_numpy(x_tr).float()
        x_va = torch.from_numpy(x_va).float()
        x_te = torch.from_numpy(x_te).float()

        # Normalize by training statistics
        mu = x_tr.mean(dim=(0, 1), keepdim=True)
        sigma = x_tr.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)

        return (x_tr - mu) / sigma, (x_va - mu) / sigma, (x_te - mu) / sigma

    def _create_windows(self, x, y, split_type):
        windows = []
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

            graph = Data(x=x_w, edge_index=self.data.edge_index, edge_label=edge_leak)

            if split_type == "train":
                if edge_leak.sum() == 0 or not self.anomaly_detection:
                    windows.append(graph)
            else:
                windows.append(graph)

        return windows

    def attach_static_features(self, *tensors):
        # normalize static features
        mu = self.data.static_features.mean(dim=0)
        sigma = self.data.static_features.std(dim=0).clamp(min=1e-8)
        self.data.static_features = (self.data.static_features - mu) / sigma

        out = []
        for x in tensors:
            static_rep = einops.repeat(self.data.static_features, 'n f -> n t f', t=x.shape[1])
            out.append(torch.cat([x, static_rep], dim=-1))
        return out

    def preprocess(self):
        # Split chronologically
        (xt, yt), (xv, yv), (xte, yte) = self._split_time()
        # Impute & normalize
        xt_norm, xv_norm, xte_norm = self._impute_and_normalize(xt, xv, xte)
        # Attach static features
        xt_norm, xv_norm, xte_norm = self.attach_static_features(xt_norm, xv_norm, xte_norm)
        # Create windows
        train_data = self._create_windows(xt_norm, yt, "train")
        val_data   = self._create_windows(xv_norm, yv, "val")
        test_data  = self._create_windows(xte_norm, yte, "test")
        return train_data, val_data, test_data
