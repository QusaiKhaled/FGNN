import copy
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
    # convert from [-1, 1] to [0, 1]
    value = (value + 1.0) / 2.0 
    
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
cached_model = load_model(ckpt_input, {**model_params, "in_channels": node_dim})

model = copy.deepcopy(cached_model)

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

test_batch = test_data[selected_sample_idx].clone()
leak_edges = torch.where(test_batch.edge_label)[0].tolist()

with c2:
    edge_to_explain = st.selectbox("Leak Edge to Explain", leak_edges)
    
col1, col2 = st.columns(2)

if col1.button("Run Explanation"):
    
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
            algorithm=algo({}),
            explanation_type="model",
            model_config=model_config,
            node_mask_type=node_mask_type,
            edge_mask_type="object",
        )

        labels = test_batch.edge_label.tolist()
        scores = model(test_batch.x, test_batch.edge_index).detach().cpu().squeeze()
        predictions = torch.sigmoid(scores) > 0.5 if output_type == "raw" else scores > 0.5
        print("predictions", predictions)
        print("scores", scores)
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
        st.dataframe(counts_df, width="stretch")
        
        print("Explaining edge:", edge_to_explain, "score:", scores[edge_to_explain], "label:", labels[edge_to_explain])

        explanation = explainer(
            x=test_batch.x,
            edge_index=test_batch.edge_index,
            index=edge_to_explain,
            # target=test_batch.edge_label,
        )

        pos_fidelity, neg_fidelity, extra = fidelity(explainer, explanation, return_extra=True)
        unfaith = unfaithfulness(explainer, explanation, top_k=0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Fidelity+", f"{pos_fidelity:.4f}")
        col2.metric("Fidelity-", f"{neg_fidelity:.4f}")
        col3.metric("Unfaithfulness", f"{unfaith:.4f}")
        
        col1, col2, col3 = st.columns(3)
        
        st.write(extra)
        
        # Display score and perturbed score
        col1.metric("Original Prediction Score", f"{scores[edge_to_explain]:.4f}")
        col2.metric("Perturbed Positive Score", f"{extra['explain_y_hat'][0, 1].item():.4f}")
        col3.metric("Perturbed Negative Score", f"{extra['complement_y_hat'][0, 1].item():.4f}")

        # 2. Coordinate Mapping
        coords_df = pd.read_excel(excel_path, index_col=0)
        coord_lookup = {
            node: (coords_df.loc[node, "X-Coord"], coords_df.loc[node, "Y-Coord"])
            for node in raw_data.node_names
        }

        # 3. Process Importance
        edge_importance_raw = explanation.edge_mask.detach().cpu()
        edge_importance = explanation.edge_mask.detach().cpu()
        edge_importance = edge_importance / (edge_importance.abs().max() + 1e-8)

        node_importance = explanation.get("node_mask")

        if node_importance is not None:
            node_importance = node_importance.detach().cpu()
            node_importance_raw = node_importance.clone()
            
            # Signed normalization to [-1, 1]
            node_importance = node_importance / (node_importance.abs().max() + 1e-8)
            
            feature_importance = node_importance.mean(dim=0)
            # use signed abs-max per node: take feature with largest absolute value (preserving its sign)
            abs_idx = node_importance.abs().argmax(dim=1)
            node_importance = node_importance[torch.arange(node_importance.size(0)), abs_idx]
            
            with st.expander("View Importance Details"):
                
                st.write("Node Importance Scores Raw")
                st.write(node_importance_raw)
                st.pyplot(node_importance_raw.plt.fig)
                
                st.write("Edge Importance Scores Raw")
                st.write(edge_importance_raw)

                st.write("Node Importance Scores Normalized")
                st.write(node_importance)
                st.dataframe(node_importance.detach().cpu().numpy())
                
                st.write("Edge Importance Scores Normalized")
                st.write(edge_importance)
                st.dataframe(edge_importance.detach().cpu().numpy())

            # plot feature importance
            st.markdown("### Feature Importance")
            fig, ax = plt.subplots()
            ax.bar(range(len(feature_importance)), feature_importance.numpy())
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Importance")
            st.pyplot(fig, width=750)
        else:
            node_importance = torch.zeros(raw_data.num_nodes)
            st.warning("No node importance computed.")

        # Button for downloading the whole sample as Excel with node and edge importance
        
        features_df = pd.DataFrame(test_batch.x.detach().cpu().numpy(), columns=[f"feat_{i}" for i in range(test_batch.x.size(1))])
        
        feature_importance_df = pd.DataFrame({
            "Feature": [f"feat_{i}" for i in range(feature_importance.size(0))],
            "Importance": feature_importance.numpy(),
        })
        
        importance_df = pd.DataFrame({
            "Node": raw_data.node_names,
            "Node Importance": node_importance.numpy(),
        })
        edge_ei = explanation.edge_index.t().tolist()
        edge_df = pd.DataFrame({
            "Edge": [f"{raw_data.node_names[u]}-{raw_data.node_names[v]}" for u, v in edge_ei],
            "Edge Importance": edge_importance.numpy(),
            "Label": labels,
            "Prediction": y_pred,
            "Score": scores,
        })
        with pd.ExcelWriter("sample.xlsx") as writer:
            features_df.to_excel(writer, sheet_name="Node Features", index=False)
            feature_importance_df.to_excel(writer, sheet_name="Feature Importance", index=False)
            importance_df.to_excel(writer, sheet_name="Node Importance", index=False)
            edge_df.to_excel(writer, sheet_name="Edge Importance", index=False)

        with open("sample.xlsx", "rb") as f:
            st.download_button(
                label="Download Importance Scores as Excel",
                data=f.read(),
                file_name="sample.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # 4. Pyvis Visualization
        cmap_text = "managua"
        
        net = Network(
            height="700px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=False,
            neighborhood_highlight=True,
        )
        net.toggle_physics(False)
        cmap = matplotlib.colormaps.get_cmap(cmap_text)
        
        # node_importance[node_importance < importance_threshold] = 0.0

        # Add Nodes
        for i, (node_imp, node) in enumerate(
            zip(node_importance.tolist(), raw_data.node_names)
        ):
            x, y = coord_lookup[node]
            if node in pressure_nodes:
                shape = "triangle"
            else:
                shape = "dot"

            color = rgba_from_cmap(cmap, node_imp, alpha=1.0)
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

        # edge_importance[edge_importance < importance_threshold] = 0.0

        for i, ((u, v), imp, lbl, pred) in enumerate(
            zip(ei, edge_importance.tolist(), labels, y_pred)
        ):

            # alpha = 0.2 + 0.8 * abs(imp)
            alpha = 1.0
            width = 5 + 5 * abs(imp)

            if i == edge_to_explain:
                color = f"rgba(255,0,255,{alpha})"  # Target Edge
            elif lbl == 1:
                color = f"rgba(255,0,0,{alpha})"  # Ground Truth Leak
            else:
                color = rgba_from_cmap(cmap, imp, alpha)

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
        st.session_state["explanation_state"] = {
            "node_importance": node_importance,
            "edge_importance": edge_importance,
            "coord_lookup": coord_lookup,
            "cmap": cmap,
            "pressure_nodes": pressure_nodes,
            "labels": labels,
            "y_pred": y_pred,
            "explanation": explanation,
            "edge_to_explain": edge_to_explain,
        }
        
if col2.button("Clear Explanation"):
    st.session_state.pop("graph_html", None)
    st.session_state.pop("explanation_state", None)
    del explainer  # Clear the explainer from memory
    del model  # Clear the model from memory
    st.success("Explanation cleared.")

# Keep the graph visible after it is generated
if "explanation_state" in st.session_state:
    explanation_state = st.session_state["explanation_state"]
    
    node_importance = explanation_state["node_importance"]
    edge_importance = explanation_state["edge_importance"]
    coord_lookup = explanation_state["coord_lookup"]
    cmap = explanation_state["cmap"]
    pressure_nodes = explanation_state["pressure_nodes"]
    labels = explanation_state["labels"]
    y_pred = explanation_state["y_pred"]
    explanation = explanation_state["explanation"]
    edge_to_explain = explanation_state["edge_to_explain"]
    
    # Plot a colormap first as legend
    fig, ax = plt.subplots(figsize=(20, 1))
    fig.subplots_adjust(bottom=0.5)
    cmap = explanation_state["cmap"]
    norm = plt.Normalize(vmin=-1, vmax=1)
    cb1 = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal"
    )
    st.pyplot(fig)

    components.html(st.session_state["graph_html"], height=750)
    
    if st.button("Save Graph PDF"):
        import networkx as nx
        import matplotlib.pyplot as plt
        
        print("node_importance", node_importance)
        print("edge_importance", edge_importance)

        # =========================
        # Build NetworkX graph
        # =========================
        G = nx.Graph()
        node_imp_nx = node_importance.clone()
        edge_imp_nx = edge_importance.clone()

        # =========================
        # Add nodes
        # =========================
        pos = {}

        for i, (imp, node) in enumerate(zip(node_imp_nx.tolist(), raw_data.node_names)):
            x, y = coord_lookup[node]
            pos[i] = (float(x), float(y))

            G.add_node(
                i,
                label=node,
                importance=imp,
                is_pressure=node in pressure_nodes,
                features=(
                    explanation.node_mask[i].detach().cpu().tolist()
                    if "node_mask" in explanation
                    else None
                ),
            )

        # =========================
        # Add edges
        # =========================
        ei = explanation.edge_index.t().tolist()

        for idx, ((u, v), imp, lbl, pred) in enumerate(
            zip(ei, edge_imp_nx.tolist(), labels, y_pred)
        ):
            G.add_edge(
                int(u),
                int(v),
                importance=abs(imp),
                signed_importance=imp,
                label=lbl,
                prediction=pred,
                is_target=(idx == edge_to_explain),
            )

        # =========================
        # Draw graph
        # =========================
        fig, ax = plt.subplots(figsize=(30, 20))
        
        print("edge importance again", edge_imp_nx)

        # ---- Nodes ----
        node_colors = [
            cmap((G.nodes[n]["importance"]+1.0) / 2.0) for n in G.nodes
        ]

        node_sizes = [80 for _ in G.nodes]

        pressure_nodes = [n for n in G.nodes if G.nodes[n]["is_pressure"]]
        normal_nodes = [n for n in G.nodes if not G.nodes[n]["is_pressure"]]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=normal_nodes,
            node_shape="o",
            node_color=[node_colors[n] for n in normal_nodes],
            node_size=80,
            ax=ax,
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=pressure_nodes,
            node_shape="^",
            node_color=[node_colors[n] for n in pressure_nodes],
            node_size=120,
            ax=ax,
        )

        # Draw labels with displacement to avoid overlap (provide explicit labels to avoid later duplicate draw)
        labels_disp = {i: G.nodes[i]["label"] for i in G.nodes}
        label_pos = {node: (x, y - 20) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels=labels_disp, font_size=8, ax=ax)

        # Clear original node label attributes so the later generic draw_networkx_labels call doesn't add a second (possibly different) label
        for i in G.nodes:
            G.nodes[i]["label"] = ""

        # ---- Edges ----
        for u, v, data in G.edges(data=True):
            sig_imp = data["signed_importance"]
            imp = data["importance"]
            
            # alpha = 0.2 + 0.8 * imp
            alpha = 1.0
            width = 1 + 5 * imp

            if data["is_target"]:
                color = (1, 0, 1, alpha)  # magenta
            elif data["label"] == 1:
                color = (1, 0, 0, alpha)  # red
            else:
                color = (cmap((sig_imp + 1.0) / 2.0))

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=width,
                edge_color=[color],
                ax=ax,
            )

        # ---- Labels (opzionali) ----
        labels = {i: G.nodes[i]["label"] for i in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

        # ---- Colorbar (stessa viridis) ----
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.01, pad=0.01)
        cbar.set_label("Importance")

        ax.axis("off")

        # =========================
        # Save PDF
        # =========================
        plt.savefig("explanation_graph.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        st.success("Graph saved as explanation_graph.pdf")