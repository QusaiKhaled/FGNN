import random
import os
import time

import pandas as pd
import torch
import numpy as np

import lovely_tensors as lt
lt.monkey_patch()

from .data import get_data
from .train_gae import GAETrainer
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
    tracker_par = params.tracker if "tracker" in params else {}

    parameters["tracker"] = {
        "project": "FGNN",
        "group": params.group,  # Main group by model name
        "tmp_dir": "tmp",
        "cache_dir": "tmp",
        **tracker_par,
    }

    wandb_run = wandb_experiment(parameters, logger, reinit=True)

    train_data, test_data = get_data(dataset_params, logger)
    node_features = train_data[0].x.shape[1]
    
    model_params["in_channels"] = node_features
    model = get_model(model_params)
    
    if "gae" in model_params.name.lower():
        gae_trainer = GAETrainer(tracker=wandb_run, logger=logger, folder=run_name)
        gae_trainer.run_gae_training(model, train_data, test_data, params)
    else:
        raise NotImplementedError(f"Model {model_params.name} is not implemented for training.")
        
    wandb_run.end()