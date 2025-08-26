import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GNNEdgeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, depth=6, dropout=0.2, message_passer="GAT", num_classes=2, **kwargs):
        super().__init__()
        self.encoder = pyg_nn.__dict__[message_passer](
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=latent_dim,
            num_layers=depth,
            dropout=dropout,
            **kwargs
        )
        self.decoder_fc1 = nn.Linear(2 * latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        x, edge_index, batch_vec = batch.x, batch.edge_index, batch.batch

        z = self.encoder(x, edge_index, batch_vec)  # [num_nodes, latent_dim]

        src = z[edge_index[0]]  # [num_edges, latent_dim]
        dst = z[edge_index[1]]  # [num_edges, latent_dim]
        edge_repr = torch.cat([src, dst], dim=1)    # [num_edges, 2 * latent_dim]

        # Inline decoder logic
        h = F.relu(self.decoder_fc1(edge_repr))
        logits = self.decoder_fc2(h)

        return logits  # shape: [num_edges, num_classes]
