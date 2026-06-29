import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_recall_curve, auc

def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Calculates Accuracy, F1 (weighted/micro/macro), Recall (weighted/micro/macro), AUC ROC, PR ROC.
    y_true: Array of binary labels (0 or 1).
    y_pred_prob: Array of predicted probabilities for class 1.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_mi = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_ma = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_mi = recall_score(y_true, y_pred, average='micro', zero_division=0)
    rec_ma = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        roc_auc = 0.0 # Handle case where only 1 class is present in batch/fold
        
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    return {
        'accuracy': acc,
        'f1_weighted': f1_w,
        'f1_micro': f1_mi,
        'f1_macro': f1_ma,
        'recall_weighted': rec_w,
        'recall_micro': rec_mi,
        'recall_macro': rec_ma,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
