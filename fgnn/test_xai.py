import os
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go


from torchmetrics import MetricCollection

from tqdm import tqdm
import time
import warnings

from fgnn.data.dataset import FastBatchDataLoader
from fgnn.explain.metrics import FidelityMetric, UnfaithfulnessMetric
from fgnn.utils.tracker import WandBTracker

warnings.filterwarnings("ignore", category=UserWarning)


class ExplainTester:
    def __init__(self, tracker: WandBTracker, logger=None, folder=None):
        self.tracker = tracker
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Running on {self.device}")
        self.folder = folder

    def evaluate(self, explainer, loader):

        bar = tqdm(loader, desc="Evaluating Explainer", unit="batch")
        
        metrics = MetricCollection({
            "fidelity": FidelityMetric(),
            "unfaithfulness": UnfaithfulnessMetric(),
        }).to(self.device)

        for i, batch in enumerate(bar):
            batch = batch.to(self.device)
            
            predictions = explainer.model(batch.x, batch.edge_index)
            positives = torch.where(predictions > 0.5)[0]
            
            explanation = explainer(
                x=batch.x,
                edge_index=batch.edge_index,
                target=batch.edge_label,
                index=positives,
            )
            
            metrics.update(explainer, explanation)
            if (i + 1) % 10 == 0:
                current_metrics = metrics.compute()
                bar.set_postfix({
                    'Pos. Fidelity': f"{current_metrics['pos_fidelity']:.4f}",
                    'Neg. Fidelity': f"{current_metrics['neg_fidelity']:.4f}",
                    'Unfaithfulness': f"{current_metrics['unfaithfulness']:.4f}",
                })

        return metrics.compute()

    def test(self, explainer, test_data, params):
        explainer.model.to(self.device)
    
        batch_size = params.get("batch_size", 32)
        test_loader = FastBatchDataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        with self.tracker.test():
            metrics = self.evaluate(explainer, test_loader)
            self.tracker.log_metrics(metrics)


