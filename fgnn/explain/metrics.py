import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.config import ExplanationType, MaskType, ModelMode, ModelReturnType

from torchmetrics import Metric
from scipy.spatial.distance import jensenshannon


def to_safe_2d(p):
    eps = 1e-9
    p = p.clamp(min=eps, max=1-eps)
    p = torch.cat([1 - p, p], dim=-1)
    p = p / p.sum(dim=-1, keepdim=True)  # normalize after clamping
    return p


def jensen_shannon_distance(p, q):
    m = 0.5 * (p + q)

    kl_pm = F.kl_div(m.log(), p, reduction="none").sum(dim=1)
    kl_qm = F.kl_div(m.log(), q, reduction="none").sum(dim=1)

    js = 0.5 * (kl_pm + kl_qm)
    js = js / math.log(2)
    # Clamp to 0 avoid numerical issues
    js = torch.clamp(js, min=0.0)
    return torch.sqrt(js)


def unfaithfulness(
    explainer: Explainer,
    explanation: Explanation,
    top_k: Optional[int] = None,
) -> float:
    r"""Evaluates how faithful an :class:`~torch_geometric.explain.Explanation`
    is to an underyling GNN predictor, as described in the
    `"Evaluating Explainability for Graph Neural Networks"
    <https://arxiv.org/abs/2208.09339>`_ paper.

    In particular, the graph explanation unfaithfulness metric is defined as

    .. math::
        \textrm{GEF}(y, \hat{y}) = 1 - \exp(- \textrm{KL}(y || \hat{y}))

    where :math:`y` refers to the prediction probability vector obtained from
    the original graph, and :math:`\hat{y}` refers to the prediction
    probability vector obtained from the masked subgraph.
    Finally, the Kullback-Leibler (KL) divergence score quantifies the distance
    between the two probability distributions.

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (Explanation): The explanation to evaluate.
        top_k (int, optional): If set, will only keep the original values of
            the top-:math:`k` node features identified by an explanation.
            If set to :obj:`None`, will use :obj:`explanation.node_mask` as it
            is for masking node features. (default: :obj:`None`)
    """
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    if top_k is not None and explainer.node_mask_type == MaskType.object:
        raise ValueError(
            "Cannot apply top-k feature selection based on a "
            "node mask of type 'object'"
        )

    node_mask = explanation.get("node_mask")
    edge_mask = explanation.get("edge_mask")
    x, edge_index = explanation.x, explanation.edge_index
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.get("prediction")
    if y is None:  # == ExplanationType.phenomenon
        y = explainer.get_prediction(x, edge_index, **kwargs)

    if node_mask is not None and top_k is not None:
        feat_importance = node_mask.sum(dim=0)
        _, top_k_index = feat_importance.topk(top_k)
        node_mask = torch.zeros_like(node_mask)
        node_mask[:, top_k_index] = 1.0

    y_hat = explainer.get_masked_prediction(
        x, edge_index, node_mask, edge_mask, **kwargs
    )

    if explanation.get("index") is not None:
        y, y_hat = y[explanation.index], y_hat[explanation.index]

    if explainer.model_config.return_type == ModelReturnType.raw:
        # Distinguish binary and multi-class classification
        if y.size(-1) == 1:
            y, y_hat = y.sigmoid(), y_hat.sigmoid()
            y = torch.cat([1 - y, y], dim=-1)
            y_hat = torch.cat([1 - y_hat, y_hat], dim=-1)
        else:
            y, y_hat = y.softmax(dim=-1), y_hat.softmax(dim=-1)
    elif explainer.model_config.return_type == ModelReturnType.log_probs:
        y, y_hat = y.exp(), y_hat.exp()
    elif explainer.model_config.return_type == ModelReturnType.probs:
        if y.size(-1) == 1:
            y = torch.cat([1 - y, y], dim=-1)
            y_hat = torch.cat([1 - y_hat, y_hat], dim=-1)

    # Ensure numerical stability
    y_hat = y_hat.clamp(min=1e-9)
    y = y.clamp(min=1e-9)

    kl_div = F.kl_div(y.log(), y_hat, reduction="batchmean")
    return 1 - float(torch.exp(-kl_div))


def fidelity(
    explainer: Explainer,
    explanation: Explanation,
    modality: str = "js",
    return_extra: bool = False,
) -> Tuple[float, float]:
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    if explainer.explanation_type == ExplanationType.phenomenon:
        raise NotImplementedError(
            "Fidelity for phenomenon explanations is not implemented yet."
        )

    node_mask = explanation.get("node_mask")
    edge_mask = explanation.get("edge_mask")
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.prediction
    
    if len(y.shape) == 1:
        y = y.unsqueeze(-1)

    explain_y_hat = explainer.get_masked_prediction(
        explanation.x,
        explanation.edge_index,
        node_mask,
        edge_mask,
        **kwargs,
    )

    complement_y_hat = explainer.get_masked_prediction(
        explanation.x,
        explanation.edge_index,
        1.0 - node_mask if node_mask is not None else None,
        1.0 - edge_mask if edge_mask is not None else None,
        **kwargs,
    )

    output_type = explainer.model_config.return_type
    if output_type == ModelReturnType.probs:
        if y.size(-1) == 1:                     
            explain_y_hat = to_safe_2d(explain_y_hat)
            complement_y_hat = to_safe_2d(complement_y_hat)
            y = to_safe_2d(y)
    elif output_type == ModelReturnType.log_probs:
        raise NotImplementedError(
            "Fidelity with 'log_probs' output type is not implemented yet."
        )
    elif output_type == "labels":
        raise NotImplementedError(
            "Fidelity with 'labels' output type is not implemented yet."
        )
    elif output_type == ModelReturnType.raw:
        if y.size(-1) == 1:
            explain_y_hat = torch.sigmoid(explain_y_hat)
            complement_y_hat = torch.sigmoid(complement_y_hat)
            y = torch.sigmoid(y)
                     
            explain_y_hat = to_safe_2d(explain_y_hat)
            complement_y_hat = to_safe_2d(complement_y_hat)
            y = to_safe_2d(y)
        else:
            explain_y_hat = torch.softmax(explain_y_hat, dim=-1)
            complement_y_hat = torch.softmax(complement_y_hat, dim=-1)
            y = torch.softmax(y, dim=-1)

    if explanation.get("index") is not None:
        y = y[explanation.index]
        explain_y_hat = explain_y_hat[explanation.index]
        complement_y_hat = complement_y_hat[explanation.index]

    if output_type == "labels":
        raise NotImplementedError(
            "Fidelity with 'labels' output type is not implemented yet."
        )
    else:
        if modality == "js":
            js = jensen_shannon_distance(explain_y_hat, y).mean()
            pos_fidelity = 1.0 - js
            
            js = jensen_shannon_distance(complement_y_hat, y).mean()
            neg_fidelity = 1.0 - js
        elif modality == "pred":
            pred_matches = (explain_y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float()
            pos_fidelity = pred_matches.mean().item()
            
            pred_matches = (complement_y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float()
            neg_fidelity = pred_matches.mean().item()
        else:
            raise ValueError(f"Unknown fidelity modality '{modality}'")
        
    if return_extra:
        extra = {
            "explain_y_hat": explain_y_hat,
            "complement_y_hat": complement_y_hat,
        }
        return float(pos_fidelity), float(neg_fidelity), extra

    return float(pos_fidelity), float(neg_fidelity)


class UnfaithfulnessMetric(Metric):
    def __init__(self, dist_sync_on_step=False, top_k: Optional[int] = None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_unfaithfulness", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.top_k = top_k

    def update(self, explainer: Explainer, explanation: Explanation):
        unfaith = unfaithfulness(explainer, explanation, top_k=self.top_k)
        self.total_unfaithfulness += unfaith
        self.count += 1

    def compute(self):
        return self.total_unfaithfulness / self.count if self.count > 0 else torch.tensor(0.0)
    
    
class FidelityMetric(Metric):
    def __init__(self, dist_sync_on_step=False, modality: str = "js"):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_pos_fidelity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_neg_fidelity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.modality = modality

    def update(self, explainer: Explainer, explanation: Explanation):
        pos_fid, neg_fid = fidelity(explainer, explanation, modality=self.modality) 
        self.total_pos_fidelity += pos_fid
        self.total_neg_fidelity += neg_fid
        self.count += 1

    def compute(self) -> Tuple[float, float]:
        if self.count == 0:
            return {"pos": torch.tensor(0.0), "neg": torch.tensor(0.0)}
        return {"pos": self.total_pos_fidelity / self.count,
                "neg": self.total_neg_fidelity / self.count}