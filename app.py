from typing import Optional, Tuple
import streamlit as st
import torch
import pandas as pd
import matplotlib.cm as cm
import streamlit.components.v1 as components
from pyvis.network import Network

from torch_geometric import explain
from torch_geometric.data import Data
from torch_geometric.explain.metric import fidelity_curve_auc
from torch_geometric.explain.config import ExplanationType, ModelMode

# Custom imports from your fgnn package
from fgnn.models import get_model
from fgnn.data import get_data
from fgnn.utils.metrics import compute_metrics
from fgnn.utils.utils import DotDict

import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.config import MaskType, ModelMode, ModelReturnType

import lovely_tensors as lt

lt.monkey_patch()


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
    
    print("Unfaithfulness Debug Info:")
    print("y:", y)
    print("y_hat:", y_hat)

    kl_div = F.kl_div(y.log(), y_hat, reduction="batchmean")
    print("KL Divergence:", kl_div)
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

    print("y:", y)
    print("explain_y_hat:", explain_y_hat)
    print("complement_y_hat:", complement_y_hat)

    if output_type == "labels":
        pos_fidelity = 1.0 - (complement_y_hat == y).float().mean()
        neg_fidelity = 1.0 - (explain_y_hat == y).float().mean()
    else:
        pos_fidelity = 1.0 - F.mse_loss(complement_y_hat, y)
        neg_fidelity = 1.0 - F.mse_loss(explain_y_hat, y)

    print("pos_fidelity:", pos_fidelity)
    print("neg_fidelity:", neg_fidelity)

    return float(pos_fidelity), float(neg_fidelity)


# --- Page Config ---
st.set_page_config(page_title="Water Network Explainer", layout="wide")


# --- Helper Functions ---
def rgba_from_cmap(cmap, value, alpha):
    r, g, b, _ = cmap(value)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


class ExplainWrapper(torch.nn.Module):
    def __init__(self, model, output_type):
        super().__init__()
        self.model = model
        self.output_type = output_type

    def forward(self, x, edge_index):
        raw = self.model(Data(x=x, edge_index=edge_index))
        if self.output_type == "raw":
            return raw
        elif self.output_type == "probs":
            if raw.size(-1) == 1:
                return torch.sigmoid(raw)
            else:
                return torch.softmax(raw, dim=-1)

@st.cache_resource
def load_data(dataset_params):

    class FakeLogger:
        def info(self, msg):
            pass

    # Load Data
    train_data, val_data, test_data, raw_data = get_data(dataset_params, FakeLogger())

    # Load Model
    node_dim = train_data[0].x.shape[1]

    return test_data, raw_data, node_dim


@st.cache_resource
def load_model(ckpt_path, model_params):
    model_params = DotDict(model_params)

    model = get_model(model_params)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
    model.eval()

    return model


CKPT_REGISTRY = {
    "checkpoints/best_model_2tee09zr.pth": (
        dict(
            name="GNN",
            hidden_dim=128,
            latent_dim=32,
            depth=8,
            dropout=0.3,
            num_classes=1,
        ),
        dict(
            window_size=12,
            stride=0.3,
            fuzzy=dict(n_clusters=3),
            return_raw=True,
        ),
    ),
    "checkpoints/best_model_nozzvx30.pth": (
        dict(
            name="GNN",
            hidden_dim=128,
            latent_dim=32,
            depth=16,
            dropout=0.3,
            num_classes=1,
        ),
        dict(
            window_size=12,
            stride=0.3,
            fuzzy=dict(n_clusters=3),
            return_raw=True,
        ),
    ),
    "checkpoints/best_model_5fb80qspl.pth": (
        dict(
            name="GNN",
            hidden_dim=128,
            latent_dim=32,
            depth=8,
            dropout=0.3,
            num_classes=1,
        ),
        dict(
            window_size=1,
            stride=1,
            fuzzy=dict(n_clusters=3),
            return_raw=True,
        ),
    ),
}

ALGORITHMS = {
        "GNNExplainer": explain.GNNExplainer(),
        "AttentionExplainer": explain.AttentionExplainer(),
        "IntegratedGradients": explain.CaptumExplainer(attribution_method="IntegratedGradients"),
        "Saliency": explain.CaptumExplainer(attribution_method="Saliency"),
        "ShapleyValueSampling": explain.CaptumExplainer(attribution_method="ShapleyValueSampling"),
        "GuidedBackprop": explain.CaptumExplainer(attribution_method="GuidedBackprop"),
        "Deconvolution": explain.CaptumExplainer(attribution_method="Deconvolution"),
        "InputXGradient": explain.CaptumExplainer(attribution_method="InputXGradient"),
    }


@st.cache_data
def load_pressure_data(pressure_csv):
    df = pd.read_csv(pressure_csv, sep=";")
    df.drop(columns=["Timestamp"], inplace=True)
    pressure_nodes = df.columns.tolist()
    return pressure_nodes


# --- Sidebar ---
st.sidebar.header("Settings")
ckpt_input = st.sidebar.selectbox("Checkpoint Path", list(CKPT_REGISTRY.keys()))
data_pt_path = st.sidebar.text_input("Data (.pt) Path", "Data/Water_Graph_dist.pt")
excel_path = st.sidebar.text_input("Coordinates Excel Path", "Data/static2.xlsx")
pressure_csv = st.sidebar.text_input(
    "Pressure CSV Path", "Data/2018_SCADA_Pressures.csv"
)
importance_threshold = st.sidebar.slider("Importance Threshold", 0.0, 1.0, 0.0)

# --- Main App ---
st.title("💧 Water Graph Leak Explainer")

model_params, dataset_params = CKPT_REGISTRY[ckpt_input]
dataset_params["path"] = data_pt_path

test_data, raw_data, node_dim = load_data(dataset_params)
pressure_nodes = load_pressure_data(pressure_csv)
model = load_model(ckpt_input, {**model_params, "in_channels": node_dim})

with st.expander("View Model & Dataset Parameters"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Parameters")
        st.json(model_params)
    with col2:
        st.subheader("Dataset Parameters")
        st.json(dataset_params)
    st.write(f"Using {node_dim}-dimensional node features.")

# Filter for samples with leaks
leak_indices = [i for i, d in enumerate(test_data) if d.edge_label.sum() > 0]

st.write("### Select Explanation Settings")
cols = st.columns(3)

# make a generator
cols = iter(cols)

with next(cols):
    selected_algorithm = st.selectbox(
        "Select Explainer Algorithm", list(ALGORITHMS.keys())
    )
with next(cols):
    node_mask_type = st.selectbox(
        "Node Mask Type", [None, "attributes", "object"]
    )
with next(cols):
    output_type = st.selectbox(
        "Explainer Output Type", ["raw", "probs"]
    )

model = ExplainWrapper(model, output_type)

c1, c2 = st.columns(2)

with c1:
    selected_sample_idx = st.selectbox("Select Test Sample", leak_indices)

test_batch = test_data[selected_sample_idx]
leak_edges = torch.where(test_batch.edge_label)[0].tolist()

with c2:
    edge_to_explain = st.selectbox("Leak Edge to Explain", leak_edges)

if st.button("Run Explanation"):
    
    algo = ALGORITHMS[selected_algorithm]
    
    with st.spinner("Computing Graph Explanation..."):
        # 1. Explainer Logic
        model_config = explain.ModelConfig(
            mode="binary_classification",
            task_level="edge",
            return_type=output_type,
        )
        explainer = explain.Explainer(
            model=model,
            algorithm=algo,
            explanation_type="model",
            model_config=model_config,
            node_mask_type=node_mask_type,
            edge_mask_type="object",
        )

        labels = test_batch.edge_label.tolist()
        scores = model(test_batch.x, test_batch.edge_index).detach().cpu().squeeze()
        predictions = torch.sigmoid(scores) > 0.5 if output_type == "raw" else scores > 0.5
        y_pred = predictions.numpy()
        y_true = test_batch.edge_label.numpy()
        scores = scores.numpy()

        # List of true positive leak edges in another expander
        true_positives_edges = [
            i
            for (i, lbl), pred in zip(enumerate(labels), y_pred)
            if lbl == 1 and pred == 1
        ]
        st.write("True Positive Leak Edges")
        st.write(true_positives_edges)

        metrics = compute_metrics(y_true, y_pred, scores)

        # Display main metrics with st.metric in columns
        st.markdown("### Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("AUC", f"{metrics['auc']:.4f}")
        col1.metric("Precision", f"{metrics['precision']:.4f}")
        col2.metric("Recall", f"{metrics['recall']:.4f}")
        col2.metric("F1 Score", f"{metrics['f1']:.4f}")
        col3.metric("Youden's J", f"{metrics['youden_j']:.4f}")

        # Detailed counts in an expander
        st.write("Detailed Confusion Counts")
        counts_df = pd.DataFrame(
            {
                "Type": ["Positive", "Negative"],
                "Total": [metrics["total_positive"], metrics["total_negative"]],
                "Correct": [metrics["correct_positive"], metrics["correct_negative"]],
                "Incorrect": [
                    metrics["incorrect_positive"],
                    metrics["incorrect_negative"],
                ],
            }
        )
        st.dataframe(counts_df, use_container_width=True)

        explanation = explainer(
            x=test_batch.x,
            edge_index=test_batch.edge_index,
            index=edge_to_explain,
            target=test_batch.edge_label,
        )

        pos_fidelity, neg_fidelity = fidelity(explainer, explanation)
        unfaith = unfaithfulness(explainer, explanation, top_k=0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Fidelity+", f"{pos_fidelity:.4f}")
        col2.metric("Fidelity-", f"{neg_fidelity:.4f}")
        col3.metric("Unfaithfulness", f"{unfaith:.4f}")

        # 2. Coordinate Mapping
        coords_df = pd.read_excel(excel_path, index_col=0)
        coord_lookup = {
            node: (coords_df.loc[node, "X-Coord"], coords_df.loc[node, "Y-Coord"])
            for node in raw_data.node_names
        }

        # 3. Process Importance
        edge_importance = explanation.edge_mask.detach().cpu()
        edge_importance = (edge_importance - edge_importance.min()) / (
            edge_importance.max() - edge_importance.min() + 1e-8
        )

        node_importance = explanation.get("node_mask")

        if node_importance is not None:
            node_importance = node_importance.detach().cpu()
            feature_importance = node_importance.mean(dim=0)
            node_importance = node_importance.mean(dim=1)
            node_importance = (node_importance - node_importance.min()) / (
                node_importance.max() - node_importance.min() + 1e-8
            )

            # plot feature importance
            st.markdown("### Feature Importance")
            fig, ax = plt.subplots()
            ax.bar(range(len(feature_importance)), feature_importance.numpy())
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Importance")
            st.pyplot(fig, width=750)
        else:
            node_importance = torch.zeros(raw_data.num_nodes)

        # 4. Pyvis Visualization
        net = Network(
            height="700px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=False,
            neighborhood_highlight=True,
        )
        net.toggle_physics(False)
        viridis = matplotlib.colormaps.get_cmap("viridis")
        
        node_importance[node_importance < importance_threshold] = 0.0

        # Add Nodes
        for i, (node_imp, node) in enumerate(
            zip(node_importance.tolist(), raw_data.node_names)
        ):
            x, y = coord_lookup[node]
            if node in pressure_nodes:
                shape = "triangle"
            else:
                shape = "dot"

            color = rgba_from_cmap(viridis, node_imp, alpha=1.0)
            feat_importance_text = "\n".join(
                [
                    f"Feat {j}: {imp:.4f}"
                    for j, imp in enumerate(
                        explanation.node_mask[i].detach().cpu().tolist()
                    )
                ]
            ) if node_mask_type is not None else ""
            net.add_node(
                i,
                label=node,
                x=float(x),
                y=-float(y),
                size=8,
                physics=False,
                color=color,
                shape=shape,
                title=f"Importance: {node_imp:.4f} \n {feat_importance_text}",
            )

        # Add Edges
        ei = explanation.edge_index.t().tolist()

        edge_importance[edge_importance < importance_threshold] = 0.0

        for i, ((u, v), imp, lbl, pred) in enumerate(
            zip(ei, edge_importance.tolist(), labels, y_pred)
        ):

            alpha = 0.2 + 0.8 * imp
            width = 1 + 5 * imp

            if i == edge_to_explain:
                color = f"rgba(255,0,255,{alpha})"  # Target Edge
            elif lbl == 1:
                color = f"rgba(255,0,0,{alpha})"  # Ground Truth Leak
            else:
                color = rgba_from_cmap(viridis, imp, alpha)

            net.add_edge(
                int(u),
                int(v),
                value=width,
                color=color,
                smooth=False,
                title=f"Importance: {imp:.4f} \n Leak: {lbl} \n Pred: {pred}",
            )

        # 5. Render HTML
        path = "temp_network.html"
        net.save_graph(path)

        with open(path, "r", encoding="utf-8") as f:
            html_data = f.read()

        st.session_state["graph_html"] = html_data

# Keep the graph visible after it is generated
if "graph_html" in st.session_state:
    # Plot a colormap first as legend
    fig, ax = plt.subplots(figsize=(20, 1))
    fig.subplots_adjust(bottom=0.5)
    cmap = matplotlib.colormaps.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=1)
    cb1 = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal"
    )
    st.pyplot(fig)

    components.html(st.session_state["graph_html"], height=750)
