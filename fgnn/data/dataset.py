import torch

import numpy as np
from torch_geometric.data import Batch


from torch_geometric.utils import negative_sampling, dropout_adj

def apply_augmentations(batch, feature_mask_rate=0.05, edge_drop_rate=0.05):
    x, edge_index = batch.x, batch.edge_index
    mask = torch.bernoulli(torch.full((x.size(1),), 1 - feature_mask_rate, device=x.device)).bool()
    augmented_x = x.clone()
    augmented_x[:, ~mask] = 0
    augmented_edge_index, _ = dropout_adj(edge_index, p=edge_drop_rate, force_undirected=False, training=True)
    return augmented_x, augmented_edge_index

# --- Optimized Data Loading with True Batching ---
class FastBatchDataLoader:
    def __init__(self, data_list, batch_size=32, shuffle=True):
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __iter__(self):
        idx = list(range(len(self.data_list)))
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            batch_data = [self.data_list[j] for j in idx[i:i + self.batch_size]]
            if not batch_data: continue
            yield Batch.from_data_list(batch_data)
    def __len__(self):
        return (len(self.data_list) + self.batch_size - 1) // self.batch_size