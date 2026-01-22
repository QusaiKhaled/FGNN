from typing import Optional
import streamlit as st
import torch
import pandas as pd
import matplotlib.cm as cm
import streamlit.components.v1 as components
from pyvis.network import Network

from torch_geometric import explain
from torch_geometric.explain.metric import fidelity_curve_auc

# Custom imports from your fgnn package
from fgnn.explain import ALGORITHMS
from fgnn.explain.metrics import unfaithfulness
from fgnn.explain.metrics import fidelity
from fgnn.models import get_model
from fgnn.data import get_data
from fgnn.explain import ExplainWrapper
from fgnn.utils.metrics import compute_metrics
from fgnn.utils.utils import DotDict

import torch

import matplotlib
import matplotlib.pyplot as plt


import lovely_tensors as lt

lt.monkey_patch()


# --- Page Config ---
st.set_page_config(page_title="Water Network Explainer", layout="wide")


# --- Helper Functions ---
def rgba_from_cmap(cmap, value, alpha):
    r, g, b, _ = cmap(value)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


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
