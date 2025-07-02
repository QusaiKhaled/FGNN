import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_max

# --- Super Efficient Batch-Optimized Sensor Attention K-Hop Aggregation ---
class BatchOptimizedSensorAttentionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, max_hops=3, value_tolerance=0.01, heads=1, dropout=0.2):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_hops = min(max_hops, 3)
        self.value_tolerance = value_tolerance
        self.heads = heads
        self.dropout = dropout
        self.node_proj = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.neighbor_proj = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.out_proj = nn.Linear(out_channels * heads, out_channels, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

    def get_k_hop_edges_batched(self, edge_index, num_nodes, k, batch_size=None):
        if k == 1:
            return edge_index
        device = edge_index.device
        current_edges = edge_index
        for hop in range(k - 1):
            adj_dict = {}
            for i in range(current_edges.shape[1]):
                src, dst = current_edges[:, i]
                src, dst = src.item(), dst.item()
                adj_dict.setdefault(src, []).append(dst)
            new_edges = set()
            for src in adj_dict:
                for inter in adj_dict[src]:
                    if inter in adj_dict:
                        for dst in adj_dict[inter]:
                            if dst != src:
                                new_edges.add((src, dst))
            if not new_edges:
                break
            new_list = list(new_edges)
            if len(new_list) > edge_index.shape[1] * (k + 1):
                idx = torch.randperm(len(new_list))[:edge_index.shape[1] * (k + 1)]
                new_list = [new_list[i] for i in idx]
            current_edges = torch.tensor(new_list, device=device).t().contiguous()
        return current_edges

    def super_efficient_batch_aggregation(self, x, edge_index, batch=None):
        device = x.device
        num_nodes = x.size(0)
        if batch is not None:
            return self.handle_batched_graphs(x, edge_index, batch)
        output = torch.zeros(num_nodes, self.out_channels * self.heads, device=device)
        for hop in range(1, self.max_hops + 1):
            hop_edges = edge_index if hop == 1 else self.get_k_hop_edges_batched(edge_index, num_nodes, hop)
            if hop_edges.shape[1] == 0:
                continue
            row, col = hop_edges
            node_feats = self.node_proj(x)
            neigh_feats = self.neighbor_proj(x)
            diffs = torch.norm(x[col] - x[row], dim=1)
            max_idx = scatter_max(diffs, row, dim=0, dim_size=num_nodes)[1]
            valid = max_idx != -1
            if valid.sum() > 0:
                valid_nodes = torch.where(valid)[0]
                mask = torch.zeros(hop_edges.shape[1], dtype=torch.bool, device=device)
                for idx in valid_nodes:
                    edges = torch.where(row == idx)[0]
                    if edges.numel() > 0:
                        e = edges[torch.argmax(diffs[edges])]
                        mask[e] = True
                sel = torch.where(mask)[0]
                r, c = row[sel], col[sel]
                combined = 0.6 * node_feats[r] + 0.4 * neigh_feats[c]
                output[r] = combined
        no_nbors = (output.sum(dim=1) == 0)
        if no_nbors.any():
            output[no_nbors] = self.node_proj(x)[no_nbors]
        return self.dropout_layer(self.out_proj(output))

    def handle_batched_graphs(self, x, edge_index, batch):
        device = x.device
        bs = batch.max().item() + 1
        outs = []
        for b in range(bs):
            mask = batch == b
            if mask.sum() == 0: continue
            nodes = torch.where(mask)[0]
            sub_e_mask = mask[edge_index[0]] & mask[edge_index[1]]
            sub_e = edge_index[:, sub_e_mask]
            mapping = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
            mapping[nodes] = torch.arange(nodes.size(0), device=device)
            sub_e = mapping[sub_e]
            sub_x = x[nodes]
            outs.append(self.super_efficient_batch_aggregation(sub_x, sub_e, batch=None))
        return torch.cat(outs, dim=0)

    def forward(self, x, edge_index, batch=None):
        return self.super_efficient_batch_aggregation(x, edge_index, batch)
    
    
# --- Decoder & Models ---
class MLPDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels * 2, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.out_layer = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, edge_index):
        row, col = edge_index
        x = torch.cat([z[row], z[col]], dim=1)
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.out_layer(x).squeeze(-1)