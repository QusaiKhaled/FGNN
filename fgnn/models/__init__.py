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

    parameters = {k:v for k,v in parameters.items() if k != 'name'}

    models = {
        'GHAE': GrassHopperAutoencoder,
        'GAE': GraphAutoencoder
    }

    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")

    return models[model_name](**parameters)