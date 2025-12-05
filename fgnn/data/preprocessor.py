from copy import deepcopy
from typing import Dict, Tuple, Union
from matplotlib.pyplot import bar
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

import einops


FractionOrTime = Union[float, int, str]


def add_drift(
    raw_data,
    node_intervals: Dict[str, Tuple[FractionOrTime, FractionOrTime]],
    step_range: tuple = (0.00005, 0.0002),
    inplace: bool = False,
    target_feature_indices: list | None = None,
    use_feature_mask: bool = False,
):
    """
    Apply incremental drift to nodes in a PyG-like Data object.

    node_intervals values can be:
      - (start_time_str, end_time_str)  -- strings parsable by pd.to_datetime
      - (start_frac, end_frac)          -- floats in [0.0, 1.0] mapping to fraction of timeline

    Other parameters:
      - step_range: range to sample per-node step (no seeding here)
      - inplace: modify input object if True, otherwise work on a deepcopy
      - target_feature_indices: list of feature indices to drift per timestep (default [0])
      - use_feature_mask: if True, and data.feature_mask exists, skip nodes with mask False
    """

    data = raw_data if inplace else deepcopy(raw_data)

    # --- node names ---
    if not hasattr(data, "node_names"):
        raise AttributeError("data must have attribute 'node_names' (list of strings).")
    node_names = list(data.node_names)

    # --- timestamps ---
    if hasattr(data, "timestamp_index") and isinstance(data.timestamp_index, pd.DatetimeIndex):
        ts_index = data.timestamp_index
    elif hasattr(data, "timestamps"):
        ts_arr = (
            data.timestamps.detach().cpu().numpy()
            if isinstance(data.timestamps, torch.Tensor)
            else np.asarray(data.timestamps)
        )
        ts_index = pd.to_datetime(ts_arr)
    else:
        raise AttributeError("data must have either 'timestamp_index' (DatetimeIndex) or 'timestamps' (int ns tensor).")

    num_timesteps = len(ts_index)
    if num_timesteps == 0:
        raise ValueError("Timestamp index is empty.")

    # --- x tensor validation and copy ---
    if not hasattr(data, "x") or not isinstance(data.x, torch.Tensor):
        raise TypeError("data.x must exist and be a torch.Tensor.")
    x = data.x.detach().cpu()
    orig_dtype = x.dtype
    x_np = x.numpy().copy()

    # --- feature mask handling (optional) ---
    feature_mask = None
    if use_feature_mask:
        if hasattr(data, "feature_mask"):
            fm = data.feature_mask.detach().cpu().numpy()
            feature_mask = fm.astype(bool) if fm.ndim == 1 else fm.any(axis=tuple(range(1, fm.ndim))).astype(bool)
        else:
            print("[Warning] use_feature_mask=True but data.feature_mask not found; continuing without mask.")
            feature_mask = None

    # --- infer layout ---
    if x_np.ndim == 3:
        num_nodes, T, num_features = x_np.shape
        if T != num_timesteps:
            raise ValueError(f"x.shape[1] ({T}) != number of timestamps ({num_timesteps}).")
        if target_feature_indices is None:
            target_feature_indices = [0]

        def apply_at(node_i, t_i, f_i, val):
            x_np[node_i, t_i, f_i] += val

    elif x_np.ndim == 2:
        num_nodes, total_features = x_np.shape
        if total_features % num_timesteps != 0:
            raise ValueError("Flattened features length not divisible by num_timesteps; cannot infer layout.")
        num_features = total_features // num_timesteps
        if target_feature_indices is None:
            target_feature_indices = [0]

        def apply_at(node_i, t_i, f_i, val):
            col = t_i * num_features + f_i
            x_np[node_i, col] += val

    else:
        raise ValueError("data.x must be 2D or 3D tensor.")

    # --- helper: interval parsing ---
    def interval_to_tpositions(start, end):
        """
        Return sorted numpy array of timestep positions (0-based) that fall into the inclusive interval.
        Accepts:
          - both numeric in [0,1] -> interpreted as fractions of timeline
          - otherwise both are parsed as datetimes via pd.to_datetime and a boolean mask is returned
        """
        # detect numeric-fraction case
        is_num_start = isinstance(start, (float, int, np.floating, np.integer))
        is_num_end = isinstance(end, (float, int, np.floating, np.integer))
        if is_num_start and is_num_end:
            s = float(start)
            e = float(end)
            if not (0.0 <= s <= 1.0 and 0.0 <= e <= 1.0):
                raise ValueError(f"Numeric interval bounds must be in [0,1], got {s}, {e}")
            if e < s:
                raise ValueError(f"Fractional interval end {e} < start {s}")

            start_idx = int(np.floor(s * num_timesteps))
            end_idx = int(np.ceil(e * num_timesteps)) - 1
            # clip
            start_idx = max(0, min(start_idx, num_timesteps - 1))
            end_idx = max(0, min(end_idx, num_timesteps - 1))
            if end_idx < start_idx:
                return np.array([], dtype=int)
            return np.arange(start_idx, end_idx + 1, dtype=int)

        # otherwise interpret as datetimes (strings or timestamps)
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        mask = (ts_index >= start_ts) & (ts_index <= end_ts)
        return np.nonzero(mask)[0]

    # --- apply drift per node ---
    for node_name, (start_val, end_val) in node_intervals.items():
        if node_name not in node_names:
            print(f"[Warning] Node '{node_name}' not found in data.node_names — skipped.")
            continue
        node_idx = node_names.index(node_name)

        if feature_mask is not None and not bool(feature_mask[node_idx]):
            print(f"[Info] Node '{node_name}' has no available features (feature_mask False); skipped.")
            continue

        t_positions = interval_to_tpositions(start_val, end_val)
        if t_positions.size == 0:
            print(f"[Info] No timesteps for node '{node_name}' within interval {start_val} - {end_val}. Skipped.")
            continue

        # sample random step for this node (no seeding here)
        step = float(np.random.uniform(step_range[0], step_range[1]))
        print(f"[Info] Node '{node_name}': applying incremental drift step={step:.6f} on {t_positions.size} timesteps")

        t_positions = np.sort(t_positions)
        for k, tpos in enumerate(t_positions):
            add_val = (k + 1) * step
            for fidx in target_feature_indices:
                apply_at(node_idx, tpos, fidx, add_val)

    # put modified array back into tensor with original dtype
    data.x = torch.from_numpy(x_np).to(dtype=orig_dtype)
    return data


# --- MODIFIED: Preprocessing for Semi-Supervised Learning ---
class SemiSupervisedPreprocessor:
    """
    This preprocessor splits time-series graph data for semi-supervised learning.
    It divides sequences into training, validation, and testing by chronological order,
    imputes missing values, normalizes features, and constructs fixed-size windows.
    """

    def __init__(
        self,
        data,
        window_size,
        stride,
        train_ratio=0.6,
        val_ratio=0.2,
        max_windows=None,
        num_nodes_features=1,
        anomaly_detection=False,
        logger=None,
    ):
        assert (
            train_ratio + val_ratio < 1.0
        ), "Train + val ratio must be < 1.0 (remainder is test)"
        self.data = data
        self.window_size = window_size

        if stride < 1:
            # Stride is a percentage of the windows_size
            stride = int(window_size * stride)

        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.max_windows = max_windows
        self.num_nodes = data.x.shape[0]
        self.num_node_features = num_nodes_features
        self.num_timesteps = data.x.shape[1] // self.num_node_features
        self.anomaly_detection = anomaly_detection
        self.logger = logger

    def _split_time(self):
        train_end = int(self.num_timesteps * self.train_ratio)
        val_end = int(self.num_timesteps * (self.train_ratio + self.val_ratio))

        seq = self.data.x.view(
            self.num_nodes, self.num_timesteps, self.num_node_features
        )

        return (
            (seq[:, :train_end, :], self.data.leak_target[:, :train_end]),
            (seq[:, train_end:val_end, :], self.data.leak_target[:, train_end:val_end]),
            (seq[:, val_end:, :], self.data.leak_target[:, val_end:]),
        )

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

        bar = tqdm(
            range(num_to_process),
            total=num_to_process,
            desc=f"Creating windows for {split_type}",
            file=getattr(self.logger, "stream", None) if self.logger else None,
        )

        for i in bar:
            start = i * self.stride
            end = start + self.window_size

            x_w = x[:, start:end, :].reshape(self.num_nodes, -1)
            y_w = y[:, start:end]

            # Vectorized edge leak computation
            node_flag = (y_w > 0).any(dim=1).float()
            edge_leak = node_flag[self.data.edge_index].max(dim=0).values

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
            static_rep = einops.repeat(
                self.data.static_features, "n f -> n t f", t=x.shape[1]
            )
            out.append(torch.cat([x, static_rep], dim=-1))
        return out

    def preprocess(self):
        # Split chronologically
        (xt, yt), (xv, yv), (xte, yte) = self._split_time()
        # Impute & normalize
        xt_norm, xv_norm, xte_norm = self._impute_and_normalize(xt, xv, xte)
        # Attach static features
        xt_norm, xv_norm, xte_norm = self.attach_static_features(
            xt_norm, xv_norm, xte_norm
        )
        # Create windows
        train_data = self._create_windows(xt_norm, yt, "train")
        val_data = self._create_windows(xv_norm, yv, "val")
        test_data = self._create_windows(xte_norm, yte, "test")
        return train_data, val_data, test_data
