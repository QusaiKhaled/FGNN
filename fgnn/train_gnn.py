import os
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go

from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, precision_recall_fscore_support, accuracy_score

from tqdm import tqdm
import time
import warnings

from fgnn.data.dataset import FastBatchDataLoader, apply_augmentations
from fgnn.loss import get_loss
from fgnn.scheduler import get_scheduler
from fgnn.utils.metrics import compute_metrics
from fgnn.utils.tracker import WandBTracker

warnings.filterwarnings("ignore", category=UserWarning)


class GNNTrainer:
    def __init__(self, tracker: WandBTracker, logger=None, folder=None):
        self.tracker = tracker
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Running on {self.device}")
        self.folder = folder

    def train_epoch(self, model, loader, opt, criterion):
        model.train()
        total_loss = 0
        bar = tqdm(loader, desc="Training", unit="batch")

        for batch in bar:
            batch = batch.to(self.device)
            opt.zero_grad()
            out = model(batch)  # should return logits for each edge
            labels = batch.edge_label
            
            # Remove class dimension from out if class == 1
            if out.size(-1) == 1:
                out = out.squeeze(-1)

            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(loader)
        self.tracker.log_metric("loss", avg_loss)
        return avg_loss

    def evaluate(self, model, loader):
        model.eval()
        all_scores, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = model(batch)  # logits
                labels = batch.edge_label

                all_scores.append(out.cpu())
                all_labels.append(labels.cpu())

        y_true = torch.cat(all_labels)
        scores = torch.cat(all_scores)
        y_pred = (torch.sigmoid(scores.squeeze(-1)) > 0.5).long() if out.size(-1) == 1 else scores.argmax(axis=1)

        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        scores = scores.numpy()

        metrics = compute_metrics(y_true, y_pred, scores)

        self.logger.info(f" Evaluation finished: AUC={metrics['auc']:.4f} | Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | Youden's J={metrics['youden_j']:.4f}")

        return metrics

    def train(self, model, train_data, val_data, test_data, params):
        
        if "training" in params:
            train_params = params.training
            sched_params = train_params.scheduler
            
            batch_size = params.training.batch_size
            epochs = params.training.epochs
            lr = params.training.learning_rate
            patience = train_params.get("patience", None)

            train_loader = FastBatchDataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = FastBatchDataLoader(val_data, batch_size=batch_size, shuffle=False)

            model = model.to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            sched = get_scheduler(opt, sched_params)

            num_classes = params.dataset.num_classes
            if "loss" not in params:
                if num_classes > 1:
                    params.loss = {"name": "BCEWithLogitsLoss"}
                else:
                    params.loss = {"name": "CrossEntropyLoss"}
            params["num_classes"] = num_classes
                    
            criterion = get_loss(params.loss)

            best_f1 = -1
            total_train_time = 0

            for epoch in range(epochs):
                start_time = time.time()
                with self.tracker.train():
                    loss = self.train_epoch(model, train_loader, opt, criterion)
                total_train_time += time.time() - start_time

                with self.tracker.validate():
                    metrics = self.evaluate(model, val_loader)
                    self.tracker.log_metrics(metrics)
                    self.tracker.log_metric("step", epoch)

                f1 = metrics['f1']
                threshold = metrics.get('threshold', 0.5)
                current_lr = opt.param_groups[0]['lr']
                sched.step(f1)
                self.logger.info(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Val F1: {f1:.4f} | LR: {current_lr:.2e}")
                if f1 > best_f1:
                    best_f1 = f1
                    self.logger.info(f"New best F1: {best_f1:.4f} saving model...")
                    torch.save(model.state_dict(), os.path.join(self.folder, 'best_model.pth'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    self.logger.info(f"Model didn't improve since {cur_patience} epochs.")
                    if patience is not None and cur_patience > patience:
                        self.logger.info(f"Early stopping triggered: {cur_patience}/{patience}")
                        break

            self.logger.info(f"Training complete — Total time: {total_train_time:.2f}s, Avg/epoch: {total_train_time/epochs:.2f}s")
            self.logger.info("Loading best model for final evaluation...")
            model.load_state_dict(torch.load(os.path.join(self.folder, 'best_model.pth'), map_location=self.device))
            model.eval()

        batch_size = params.test.get("batch_size", params.get("training", {}).get("batch_size", None))
        test_loader = FastBatchDataLoader(test_data, batch_size=batch_size, shuffle=False)
        model.eval()
        model = model.to(self.device)
        
        with self.tracker.test():
            metrics = self.evaluate(model, test_loader)
            self.tracker.log_metrics(metrics)
