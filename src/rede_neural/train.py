import argparse
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import os

from data_ingestion import load_data
from preprocess import preprocess_and_split
from model import MLP
from utils import plot_learning_curves, plot_confusion_matrix, compute_permutation_importance, plot_feature_importance
from evaluate import evaluate_model
from sklearn.metrics import accuracy_score

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / total

def main(learning_rate, hidden_size, num_epochs):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Ingestion and Preprocessing
    X_raw, y_raw, feature_names, target_names = load_data()
    train_data, val_data, test_data, scaler = preprocess_and_split(X_raw, y_raw, device)
    
    X_train_tensor, y_train_tensor = train_data
    X_val_tensor, y_val_tensor = val_data
    X_test_tensor, y_test_tensor = test_data

    input_size = X_train_tensor.shape[1]
    num_classes = len(set(y_raw))
    
    # Model Setup
    model = MLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # MLflow Tracking
    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "num_epochs": num_epochs,
            "input_size": input_size,
            "num_classes": num_classes
        })

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        print("Iniciando treinamento...")
        for epoch in range(num_epochs):
            model.train()
            train_outputs = model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                
                train_acc = calculate_accuracy(train_outputs, y_train_tensor)
                val_acc = calculate_accuracy(val_outputs, y_val_tensor)
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
            
            # Log metrics every epoch
            mlflow.log_metrics({
                "train_loss": train_loss.item(),
                "val_loss": val_loss.item(),
                "train_acc": train_acc,
                "val_acc": val_acc
            }, step=epoch)

        # Evaluation
        metrics, cm = evaluate_model(model, X_test_tensor, y_test_tensor, num_classes)
        mlflow.log_metrics(metrics)

        print("\nTreinamento e avaliacao concluidos.")
        print(f"Metricas de teste: {metrics}")

        # Artifacts
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        curves_path = plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, artifacts_dir)
        mlflow.log_artifact(curves_path)
        
        cm_path = plot_confusion_matrix(cm, list(target_names), artifacts_dir)
        mlflow.log_artifact(cm_path)
        
        # Feature Importance (Permutation)
        def score_fn(m, X, y):
            with torch.no_grad():
                out = m(X)
                _, pred = torch.max(out.data, 1)
            return accuracy_score(y.cpu().numpy(), pred.cpu().numpy())
            
        importances = compute_permutation_importance(model, X_val_tensor, y_val_tensor, score_fn)
        fi_path = plot_feature_importance(importances, feature_names, artifacts_dir)
        mlflow.log_artifact(fi_path)

        # Log Model
        from mlflow.models.signature import infer_signature
        with torch.no_grad():
             signature = infer_signature(X_train_tensor.cpu().numpy(), model(X_train_tensor).cpu().numpy())
             
        mlflow.pytorch.log_model(model, "model", signature=signature)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=200)
    args = parser.parse_args()
    
    main(args.learning_rate, args.hidden_size, args.num_epochs)
