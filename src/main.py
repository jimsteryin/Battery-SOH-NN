import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import load_all_data, create_dataset, create_sequence_dataset, prepare_training_data
from model import SOHNet, SOHGRU

# Configuration
DATA_DIR = "../data/RawData"
CHECKPOINT_DIR = "../checkpoints"
RESULTS_DIR = "../results"
TEST_BATS_PATH = "../test_batteries.txt"

def train_mlp(args):
    print("=== Training MLP (SOHNet) ===")
    
    # 1. Load Data
    all_cycles = load_all_data(DATA_DIR)
    
    # 2. Split
    batteries = list(set(c['battery_id'] for c in all_cycles))
    batteries.sort()
    
    if os.path.exists(TEST_BATS_PATH):
        with open(TEST_BATS_PATH, 'r') as f:
            test_bats = [line.strip() for line in f.readlines()]
        train_bats = [b for b in batteries if b not in test_bats]
    else:
        random.seed(42)
        random.shuffle(batteries)
        split_idx = int(len(batteries) * 0.8)
        train_bats = batteries[:split_idx]
        test_bats = batteries[split_idx:]
        
    train_cycles = [c for c in all_cycles if c['battery_id'] in train_bats]
    test_cycles = [c for c in all_cycles if c['battery_id'] in test_bats]
    
    # 3. Preprocess
    print("Creating Snippets (5% Coverage)...")
    train_snippets, y_train = create_dataset(train_cycles, min_duration=10, max_duration=10, coverage=0.05)
    test_snippets, y_test = create_dataset(test_cycles, min_duration=10, max_duration=10, coverage=0.05)
    
    print(f"Train samples: {len(train_snippets)}")
    print(f"Test samples: {len(test_snippets)}")
    
    # 4. Features & Scaling
    X_train_df = prepare_training_data(train_snippets)
    X_test_df = prepare_training_data(test_snippets)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
    
    # 5. Training
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    model = SOHNet(X_train_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = args.epochs
    batch_size = 64
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.6f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "soh_model_nn.pth"))
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break
                
    # 6. Eval
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "soh_model_nn.pth")))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
        
    r2 = r2_score(y_test, y_pred)
    print(f"Final R2 Score: {r2:.4f}")
    print(f"Model saved to {os.path.join(CHECKPOINT_DIR, 'soh_model_nn.pth')}")


def main():
    parser = argparse.ArgumentParser(description="Battery SOH Training")
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    args = parser.parse_args()
    
    # Ensure dirs exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    train_mlp(args)

if __name__ == "__main__":
    main()
