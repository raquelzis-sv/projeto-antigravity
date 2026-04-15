import mlflow
import mlflow.sklearn
from utils import setup_logging

def train_model():
    """
    Script principal de treino com tracking do MLflow.
    """
    setup_logging()
    
    with mlflow.start_run():
        # TODO: Implementar treinamento
        print("Iniciando treinamento...")
        pass

if __name__ == "__main__":
    train_model()
