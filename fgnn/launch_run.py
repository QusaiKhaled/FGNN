import random
import os
import time

import pandas as pd
import torch
import numpy as np

import lovely_tensors as lt

from fgnn.train_gnn import GNNTrainer
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
    anomaly_detection = "gae" in model_params.name.lower()
    dataset_params["anomaly_detection"] = anomaly_detection

    wandb_run = wandb_experiment(parameters, logger, reinit=True)

    train_data, val_data, test_data = get_data(dataset_params, logger)

    num_classes = max([int(t.edge_label.max().item()) for t in train_data])
    if num_classes > 1:
        num_classes += 1  # include zero class
    node_features = train_data[0].x.shape[1]

    dataset_params["num_classes"] = num_classes

    model_params["in_channels"] = node_features
    model_params["num_classes"] = num_classes
    model = get_model(model_params)
    
    if anomaly_detection:
        gae_trainer = GAETrainer(tracker=wandb_run, logger=logger, folder=run_name)
        gae_trainer.train(model, train_data, val_data, test_data, params)
    else:
        gnn_trainer = GNNTrainer(tracker=wandb_run, logger=logger, folder=run_name)
        gnn_trainer.train(model, train_data, val_data, test_data, params)

    wandb_run.end()