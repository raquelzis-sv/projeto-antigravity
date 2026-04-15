import argparse
import os
import yaml
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

from rede_neural.data_ingestion import load_data
from rede_neural.preprocess import preprocess_and_split
from rede_neural.model import MLP
from rede_neural.evaluate import evaluate_model
from rede_neural.utils import (
    plot_learning_curves,
    plot_confusion_matrix,
    compute_permutation_importance,
    plot_feature_importance,
)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / total


def main(config_path):
    config = load_config(config_path)

    learning_rate = config["learning_rate"]
    hidden_size = config["hidden_size"]
    num_epochs = config["num_epochs"]
    experiment_name = config["experiment_name"]
    run_name = config["run_name"]
    question = config.get("question", "Sem pergunta definida")

    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_raw, y_raw, feature_names, target_names = load_data()
    train_data, val_data, test_data, scaler = preprocess_and_split(X_raw, y_raw, device)

    X_train_tensor, y_train_tensor = train_data
    X_val_tensor, y_val_tensor = val_data
    X_test_tensor, y_test_tensor = test_data

    input_size = X_train_tensor.shape[1]
    num_classes = len(set(y_raw))

    model = MLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "dataset": "iris",
            "model_type": "MLP",
            "question": question
        })

        mlflow.log_params({
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "num_epochs": num_epochs,
            "input_size": input_size,
            "num_classes": num_classes,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss"
        })

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            train_acc = calculate_accuracy(outputs, y_train_tensor)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_acc = calculate_accuracy(val_outputs, y_val_tensor)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            mlflow.log_metric("val_loss", val_loss.item(), step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        metrics, cm = evaluate_model(model, X_test_tensor, y_test_tensor, num_classes)
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1_score"]
        })

        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        learning_curve_path = plot_learning_curves(
            train_losses, val_losses, train_accuracies, val_accuracies, artifacts_dir
        )
        confusion_matrix_path = plot_confusion_matrix(
            cm, list(target_names), artifacts_dir
        )

        from sklearn.metrics import accuracy_score
        
        def score_fn(m, X, y):
            with torch.no_grad():
                out = m(X)
                _, pred = torch.max(out.data, 1)
            return accuracy_score(y.cpu().numpy(), pred.cpu().numpy())

        importance_scores = compute_permutation_importance(
            model, X_test_tensor, y_test_tensor, score_fn
        )
        feature_importance_path = plot_feature_importance(
            importance_scores, list(feature_names), artifacts_dir
        )

        mlflow.log_artifact(learning_curve_path)
        mlflow.log_artifact(confusion_matrix_path)
        mlflow.log_artifact(feature_importance_path)

        mlflow.pytorch.log_model(model, "model")

        print("Treinamento concluído com sucesso.")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1-Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho para o arquivo YAML de configuração"
    )
    args = parser.parse_args()
    main(args.config)