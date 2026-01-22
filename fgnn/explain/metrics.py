from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.config import ExplanationType, MaskType, ModelMode, ModelReturnType

from torchmetrics import Metric


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
    output_type: str = "probs",
) -> Tuple[float, float]:
    r"""Evaluates the fidelity of an
    :class:`~torch_geometric.explain.Explainer` given an
    :class:`~torch_geometric.explain.Explanation`, as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    Fidelity evaluates the contribution of the produced explanatory subgraph
    to the initial prediction, either by giving only the subgraph to the model
    (fidelity-) or by removing it from the entire graph (fidelity+).
    The fidelity scores capture how good an explainable model reproduces the
    natural phenomenon or the GNN model logic.

    For **phenomenon** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = y_i) \|

        \textrm{fid}_{-} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_S} = y_i) \|

    For **model** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = \hat{y}_i)

        \textrm{fid}_{-} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_S} = \hat{y}_i)

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (Explanation): The explanation to evaluate.
    """
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    if explainer.explanation_type == ExplanationType.phenomenon:
        raise NotImplementedError(
            "Fidelity for phenomenon explanations is not implemented yet."
        )

    node_mask = explanation.get("node_mask")
    edge_mask = explanation.get("edge_mask")
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.target

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

    if output_type == "probs":
        explain_y_hat = torch.sigmoid(explain_y_hat)
        complement_y_hat = torch.sigmoid(complement_y_hat)
    elif output_type == "log_probs":
        explain_y_hat = torch.log_softmax(explain_y_hat, dim=-1)
        complement_y_hat = torch.log_softmax(complement_y_hat, dim=-1)
    elif output_type == "labels":
        explain_y_hat = explainer.get_target(explain_y_hat)
        complement_y_hat = explainer.get_target(complement_y_hat)

    if explanation.get("index") is not None:
        y = y[explanation.index]
        explain_y_hat = explain_y_hat[explanation.index]
        complement_y_hat = complement_y_hat[explanation.index]

    if output_type == "labels":
        pos_fidelity = 1.0 - (complement_y_hat == y).float().mean()
        neg_fidelity = 1.0 - (explain_y_hat == y).float().mean()
    else:
        pos_fidelity = 1.0 - F.mse_loss(complement_y_hat, y)
        neg_fidelity = 1.0 - F.mse_loss(explain_y_hat, y)

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
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_pos_fidelity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_neg_fidelity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, explainer: Explainer, explanation: Explanation):
        pos_fid, neg_fid = fidelity(explainer, explanation)
        self.total_pos_fidelity += pos_fid
        self.total_neg_fidelity += neg_fid
        self.count += 1

    def compute(self) -> Tuple[float, float]:
        if self.count == 0:
            return {"pos_fidelity": torch.tensor(0.0), "neg_fidelity": torch.tensor(0.0)}
        return {"pos_fidelity": self.total_pos_fidelity / self.count,
                "neg_fidelity": self.total_neg_fidelity / self.count}