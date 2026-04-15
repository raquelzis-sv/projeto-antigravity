import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

def evaluate_model(model, X_test, y_test, num_classes=3):
    """Avalia o modelo e retorna as métricas."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        
    y_test_np = y_test.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    probs_np = torch.softmax(outputs, dim=1).cpu().numpy()
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test_np, predicted_np)
    metrics['precision'] = precision_score(y_test_np, predicted_np, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_test_np, predicted_np, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_test_np, predicted_np, average='weighted', zero_division=0)
    
    # ROC AUC (One-vs-Rest)
    y_test_bin = label_binarize(y_test_np, classes=list(range(num_classes)))
    if num_classes == 2:
        metrics['roc_auc'] = roc_auc_score(y_test_np, probs_np[:, 1])
    else:
        metrics['roc_auc'] = roc_auc_score(y_test_bin, probs_np, multi_class='ovr')
        
    cm = confusion_matrix(y_test_np, predicted_np)
    
    return metrics, cm
