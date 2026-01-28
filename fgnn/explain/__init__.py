import torch
from torch_geometric import explain
from torch_geometric.data import Data


ALGORITHMS = {
        "GNNExplainer": lambda params: explain.GNNExplainer(**params),
        "AttentionExplainer": lambda params: explain.AttentionExplainer(**params),
        "IntegratedGradients": lambda params: explain.CaptumExplainer(attribution_method="IntegratedGradients", **params),
        "Saliency": lambda params: explain.CaptumExplainer(attribution_method="Saliency", **params),
        "ShapleyValueSampling": lambda params: explain.CaptumExplainer(attribution_method="ShapleyValueSampling", **params),
        "GuidedBackprop": lambda params: explain.CaptumExplainer(attribution_method="GuidedBackprop", **params),
        "Deconvolution": lambda params: explain.CaptumExplainer(attribution_method="Deconvolution", **params),
        "InputXGradient": lambda params: explain.CaptumExplainer(attribution_method="InputXGradient", **params),
    }


def get_explainer(model, params):
    algo = params["algorithm"]
    output_type = params["output_type"]
    model.train()
    
    algo_params = params["algo_params"] if "algo_params" in params else {}
    node_mask_type = params.get("node_mask_type", "attributes")
    
    model_config = explain.ModelConfig(
        mode="binary_classification",
        task_level="edge",
        return_type=output_type,
    )
    explainer = explain.Explainer(
        model=ExplainWrapper(model, output_type),
        algorithm=ALGORITHMS[algo](algo_params),
        explanation_type="model",
        model_config=model_config,
        node_mask_type=node_mask_type,
        edge_mask_type="object",
    )
    return explainer


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