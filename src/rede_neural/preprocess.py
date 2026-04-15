from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

def preprocess_and_split(X, y, device):
    """Normaliza e divide os dados (respeitando Train60/Val20/Test20)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 80/20 split
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    # 75/25 of the remaining 80% -> 60/20
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    # Converter para tensores
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    return (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), (X_test_tensor, y_test_tensor), scaler
