import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_dir="artifacts"):
    """Salva gráficos de loss e acurácia."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Treino')
    plt.plot(val_losses, label='Validação')
    plt.title('Loss durante o treinamento')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Treino')
    plt.plot(val_accuracies, label='Validação')
    plt.title('Acurácia durante o treinamento')
    plt.xlabel('Epoch')
    plt.ylabel('Acurácia (%)')
    plt.legend()

    plt.tight_layout()
    file_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(file_path)
    plt.close()
    return file_path

def plot_confusion_matrix(cm, classes, save_dir="artifacts"):
    """Gera e salva gráfico da matriz de confusão."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(file_path)
    plt.close()
    return file_path

def compute_permutation_importance(model, X_val, y_val, metric_fn):
    """Calcula a Permutation Importance para o modelo PyTorch dado um tensor."""
    model.eval()
    baseline_score = metric_fn(model, X_val, y_val)
    importances = []
    
    X_val_np = X_val.cpu().numpy()
    
    for i in range(X_val_np.shape[1]):
        X_val_permuted = X_val_np.copy()
        np.random.shuffle(X_val_permuted[:, i])
        
        X_val_perm_tensor = torch.FloatTensor(X_val_permuted).to(X_val.device)
        permuted_score = metric_fn(model, X_val_perm_tensor, y_val)
        
        importances.append(baseline_score - permuted_score)
        
    return np.array(importances)

def plot_feature_importance(importances, feature_names, save_dir="artifacts"):
    """Gera gráfico da importância de variáveis."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importância Relativa (Permutation)')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(file_path)
    plt.close()
    return file_path
