import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# LOSS DEFINITIONS
# =============================

class WeightedBCEWithLogits(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight
        )


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        pt = torch.where(targets.bool(), p, 1 - p)
        w = self.alpha * torch.where(targets.bool(), torch.ones_like(p), 1 - self.alpha)
        loss = - w * (1 - pt).pow(self.gamma) * torch.log(pt.clamp_min(self.eps))
        return loss.mean()


class PairwiseLogisticRankLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_pos, s_neg):
        # expects logits for positives and negatives separately
        diff = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)
        return F.softplus(-diff).mean()


class GraphTVL1Loss(nn.Module):
    def __init__(self, lmb_tv=1.0, lmb_l1=0.001):
        super().__init__()
        self.lmb_tv = lmb_tv
        self.lmb_l1 = lmb_l1

    def forward(self, probs, edge_index, edge_weight=None):
        src, dst = edge_index
        if edge_weight is None:
            tv = (probs[src] - probs[dst]).abs().mean()
        else:
            tv = (edge_weight * (probs[src] - probs[dst]).abs()).mean()
        l1 = probs.mean()
        return self.lmb_tv * tv + self.lmb_l1 * l1


# =============================
# FACTORY FUNCTION
# =============================

def get_loss(parameters):
    """
    Factory function to get a loss by name.

    Args:
        parameters (dict): Dictionary containing loss parameters. Must include 'name' key.

    Returns:
        An instance of the requested loss.

    Raises:
        ValueError: If the loss name is not recognized.
    """

    loss_name = parameters["name"]
    params = {k: v for k, v in parameters.items() if k != "name"}

    losses = {
        "WeightedBCE": WeightedBCEWithLogits,
        "Focal": FocalLossWithLogits,
        "Rank": PairwiseLogisticRankLoss,
        "GraphTVL1": GraphTVL1Loss,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "CrossEntropyLoss": nn.CrossEntropyLoss
    }

    if loss_name not in losses:
        raise ValueError(f"Unknown loss name: {loss_name}")

    return losses[loss_name](**params)

