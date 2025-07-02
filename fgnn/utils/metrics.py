import numpy as np
from sklearn import metrics

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    thresholds[np.isinf(thresholds)] = 1.0

    output = {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}

    output = {k:v.item() if isinstance(v,np.float64) or isinstance(v,np.float32) else v for k,v in output.items()}

    return output


def compute_imagewise_f1_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels):

    y_pred = np.rint(anomaly_prediction_weights)
    precision_score = metrics.precision_score(anomaly_ground_truth_labels,y_pred)
    recall_score = metrics.recall_score(anomaly_ground_truth_labels,y_pred)
    f1_score = 2*recall_score*precision_score/(recall_score+precision_score+1e-6)
    
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2*recall*precision/(recall+precision+1e-6)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    best_f1_precision = precision[np.argmax(f1_scores)]
    best_f1_recall = recall[np.argmax(f1_scores)]

    output = {"f1":f1_score, "precision":precision_score, "recall":recall_score,
    "best_threshold": best_threshold, "best_f1": best_f1, "best_f1_precision": best_f1_precision, "best_f1_recall": best_f1_recall}
    
    output = {k:v.item() if isinstance(v,np.float64) or isinstance(v,np.float32) else v for k,v in output.items()}
    
    return output

def compute_imagewise_f1_metrics_test(anomaly_prediction_weights, anomaly_ground_truth_labels):

    y_pred = np.rint(anomaly_prediction_weights)
    precision_score = metrics.precision_score(anomaly_ground_truth_labels,y_pred)
    recall_score = metrics.recall_score(anomaly_ground_truth_labels,y_pred)
    f1_score = 2*recall_score*precision_score/(recall_score+precision_score+1e-6)

    output = {"f1":f1_score, "precision":precision_score, "recall":recall_score}

    output = {k:v.item() if isinstance(v,np.float64) or isinstance(v,np.float32) else v for k,v in output.items()}  

    return output

def compute_imagewise_f1_metrics_test_best(anomaly_prediction_weights, anomaly_ground_truth_labels):

    y_pred = np.rint(anomaly_prediction_weights)
    precision_score = metrics.precision_score(anomaly_ground_truth_labels,y_pred)
    recall_score = metrics.recall_score(anomaly_ground_truth_labels,y_pred)
    f1_score = 2*recall_score*precision_score/(recall_score+precision_score+1e-6)

    output = {"best_thresh_f1":f1_score, "best_thresh_precision":precision_score, "best_thresh_recall":recall_score}

    output = {k:v.item() if isinstance(v,np.float64) or isinstance(v,np.float32) else v for k,v in output.items()}  

    return output