import os
import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, precision_recall_fscore_support

from tqdm import tqdm
import time
import warnings

from fgnn.data.dataset import FastBatchDataLoader, apply_augmentations
from fgnn.utils.tracker import WandBTracker

warnings.filterwarnings("ignore", category=UserWarning)


class GAETrainer:
    def __init__(self, tracker: WandBTracker, logger=None, folder=None):
        self.tracker = tracker
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.folder = folder

    # --- Training & Evaluation Logic for GAE ---
    def train_gae_epoch(self, model, loader, opt):
        model.train()
        total_loss = 0
        bar = tqdm(loader, desc="Training GAE", unit="batch")
        
        for batch in bar:
            batch = batch.to(self.device)
            opt.zero_grad()
            augmented_x, augmented_edge_index = apply_augmentations(batch)
            z = model.encode(augmented_x, augmented_edge_index, batch.batch)
            pos_edge_index = batch.edge_index
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes=batch.num_nodes, num_neg_samples=pos_edge_index.size(1))
            pos_logits = model.decode(z, pos_edge_index)
            neg_logits = model.decode(z, neg_edge_index)
            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([torch.ones(pos_logits.size(0)), torch.zeros(neg_logits.size(0))], dim=0).to(self.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            bar.set_postfix({'loss': loss.item()})
        return total_loss / len(loader)

    def evaluate_gae(self, model, loader, threshold=0.5, epoch=None):
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                z = model.encode(batch.x, batch.edge_index, batch.batch)
                logits = model.decode(z, batch.edge_index)
                anomaly_scores = -torch.log(torch.sigmoid(logits) + 1e-8)
                labels = batch.edge_leak_target
                if anomaly_scores.shape[0] != labels.shape[0]:
                    m = min(anomaly_scores.shape[0], labels.shape[0])
                    anomaly_scores = anomaly_scores[:m]
                    labels = labels[:m]
                all_scores.append(anomaly_scores.cpu())
                all_labels.append(labels.cpu().long())
        if not all_scores:
            return {}
        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()
        preds = (scores > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        fpr, tpr, thr = roc_curve(labels, scores)
        tp = ((preds==1)&(labels==1)).sum(); fn = ((preds==0)&(labels==1)).sum()
        tn = ((preds==0)&(labels==0)).sum(); fp = ((preds==1)&(labels==0)).sum()
        auc = roc_auc_score(labels, scores) if len(np.unique(labels))>=2 else 0.5
        
        # precision recall curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # avoid divide-by-zero
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        
        youden_j = tpr.max() - fpr.max()
        self.logger.info(f"Threshold={threshold:.4f} | AUC={auc:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f} | Youden's J={youden_j:.4f}")

        return {'auc': auc, 'total_positive': int((labels==1).sum()), 'correct_positive': int(tp), 'incorrect_positive': int(fn), 'total_negative': int((labels==0).sum()), 'correct_negative': int(tn), 'incorrect_negative': int(fp), 'precision': precision, 'recall': recall, 'f1': f1, 'youden_j': youden_j, 'threshold': best_threshold}

    def run_gae_training(self, gae, train_data, test_data, params):
        
        train_params = params.training
        
        batch_size = train_params.batch_size
        epochs = train_params.epochs
        lr = train_params.learning_rate
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        train_loader = FastBatchDataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = FastBatchDataLoader(test_data, batch_size=batch_size, shuffle=False)
        model = gae.to(device)
        self.logger.info(f"GAE model params: {sum(p.numel() for p in model.parameters()):,}")
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=10, factor=0.7)
        best_auc = 0
        best_threshold = None
        total_train_time = 0
        for epoch in range(epochs):
            start_time = time.time()
            with self.tracker.train():
                loss = self.train_gae_epoch(model, train_loader, opt)
            total_train_time += time.time() - start_time
            with self.tracker.validate():
                metrics = self.evaluate_gae(model, test_loader, epoch+1)
                self.tracker.log_metrics(metrics)
                self.tracker.log_metric("step", epoch)
            auc = metrics['auc']
            threshold = metrics.get('threshold', 0.5)
            current_lr = opt.param_groups[0]['lr']
            sched.step(auc)
            self.logger.info(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Test AUC: {auc:.4f} | LR: {current_lr:.2e}")
            if auc > best_auc:
                best_auc = auc
                best_threshold = threshold
                self.logger.info(f"New best AUC: {best_auc:.4f} saving model...")
                torch.save(model.state_dict(), os.path.join(self.folder, 'best_gae_model.pth'))

        self.logger.info(f"\nTotal training time: {total_train_time:.2f}s, Avg per epoch: {total_train_time/epochs:.2f}s")
        self.logger.info("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load(os.path.join(self.folder, 'best_gae_model.pth'), map_location=device))
        model.eval()
        with self.tracker.test():
            self.evaluate_gae(model, test_loader, threshold=best_threshold)