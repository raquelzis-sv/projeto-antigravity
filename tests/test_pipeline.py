import sys
import os
import torch
import pytest

# Adiciona src ao path para os testes encontrarem os modulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'rede_neural')))

from data_ingestion import load_data
from preprocess import preprocess_and_split
from model import MLP

def test_data_ingestion():
    X, y, features, classes = load_data()
    assert X.shape[0] == 150
    assert X.shape[1] == 4
    assert len(features) == 4
    assert len(classes) == 3

def test_preprocessing():
    X, y, _, _ = load_data()
    device = torch.device('cpu')
    train, val, test, scaler = preprocess_and_split(X, y, device)
    
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    # 60/20/20 of 150 is 90/30/30
    assert X_train.shape[0] == 90
    assert X_val.shape[0] == 30
    assert X_test.shape[0] == 30
    assert isinstance(X_train, torch.Tensor)

def test_model_forward():
    input_size = 4
    hidden_size = 10
    num_classes = 3
    model = MLP(input_size, hidden_size, num_classes)
    
    dummy_input = torch.randn(5, input_size)
    output = model(dummy_input)
    
    assert output.shape == (5, num_classes)
