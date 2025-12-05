import torch
from fgnn.models.GNN import GNNEdgeClassifier
from .GAE import GrassHopperAutoencoder, GraphAutoencoder


def get_model(parameters):
    """
    Factory function to get a model by name.

    Args:
        parameters (dict): Dictionary containing model parameters. Must include 'name' key.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If the model_name is not recognized.
    """
    
    model_name = parameters.name
    checkpoint = parameters.get("checkpoint", None)

    parameters = {k:v for k,v in parameters.items() if k not in ['name', "checkpoint"]}

    models = {
        'GHAE': GrassHopperAutoencoder,
        'GAE': GraphAutoencoder,
        'GNN': GNNEdgeClassifier,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")

    model = models[model_name](**parameters)
    
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    return model