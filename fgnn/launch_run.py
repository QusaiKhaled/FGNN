import random
import os
import time

import pandas as pd
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt

from .data import get_data
from .train_gae import run_gae_training
from .utils.utils import DotDict
from .utils.tracker import wandb_experiment
from .models import get_model
from .utils.logger import get_logger


def launch_run(parameters, run_name):

    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    logger = get_logger(run_name)
    os.makedirs(run_name, exist_ok=True)

    params = DotDict(parameters)
    dataset_params = params.dataset
    model_params = params.model

    parameters["tracker"] = {
        "project": "FGNN",
        "group": params.group,  # Main group by model name
    }

    wandb_run = wandb_experiment(parameters, logger, reinit=True)

    train_data, test_data = get_data(dataset_params, logger)
    node_features = train_data[0].x.shape[1]
    
    model_params["in_channels"] = node_features
    model = get_model(model_params)
    
    if "gae" in model_params.name.lower():
        run_gae_training(model, train_data, test_data, params, logger, wandb_run, run_name)
    else:
        raise NotImplementedError(f"Model {model_params.name} is not implemented for training.")
        
    wandb_run.end()