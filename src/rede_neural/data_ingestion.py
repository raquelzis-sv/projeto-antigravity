from sklearn.datasets import load_iris

def load_data():
    """Carrega o dataset Iris."""
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y, iris.feature_names, iris.target_names
