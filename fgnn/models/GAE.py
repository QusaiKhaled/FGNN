import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter_max

from .modules import BatchOptimizedSensorAttentionConv, MLPDecoder


class GrassHopperAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, depth=6, dropout=0.2, max_hops=3):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else hidden_dim
            out_ch = latent_dim if i == depth - 1 else hidden_dim
            self.convs.append(BatchOptimizedSensorAttentionConv(in_ch, out_ch, max_hops=max_hops))
        self.skip_proj = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = in_channels if i == 0 else hidden_dim
            out_ch = hidden_dim
            if in_ch == out_ch:
                self.skip_proj.append(nn.Identity())
            else:
                self.skip_proj.append(nn.Linear(in_ch, out_ch))
        self.decoder = MLPDecoder(in_channels=latent_dim, hidden_channels=hidden_dim)

    def encode(self, x, edge_index, batch):
        prev = x
        for i, conv in enumerate(self.convs):
            h = conv(prev, edge_index, batch)
            if i < self.depth - 1:
                h = F.elu(h)
                h = self.dropout(h)
                skip = self.skip_proj[i](prev)
                h = h + skip
            prev = h
        return prev

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)
    
    
class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, depth=6, dropout=0.2, message_passer="GAT",  **kwargs):
        super().__init__()
        self.encoder = pyg_nn.__dict__[message_passer](
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=latent_dim,
            num_layers=depth,
            dropout=dropout,
            **kwargs
        )
        self.decoder = MLPDecoder(in_channels=latent_dim, hidden_channels=hidden_dim)

    def encode(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)