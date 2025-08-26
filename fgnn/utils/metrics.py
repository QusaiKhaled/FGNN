import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, scores):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    fpr, tpr, thr = roc_curve(y_true, scores)
    tp = ((y_pred==1)&(y_true==1)).sum(); fn = ((y_pred==0)&(y_true==1)).sum()
    tn = ((y_pred==0)&(y_true==0)).sum(); fp = ((y_pred==1)&(y_true==0)).sum()
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true))>=2 else 0.5
    
    # precision recall curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # avoid divide-by-zero
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    youden_j = tpr.max() - fpr.max()

    return {'auc': auc, 'total_positive': int((y_true==1).sum()), 'correct_positive': int(tp), 'incorrect_positive': int(fn), 'total_negative': int((y_true==0).sum()), 'correct_negative': int(tn), 'incorrect_negative': int(fp), 'precision': precision, 'recall': recall, 'f1': f1, 'youden_j': youden_j, 'threshold': best_threshold}