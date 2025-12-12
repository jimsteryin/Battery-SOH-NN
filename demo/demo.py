import sys
import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import SOHNet
from utils import load_all_data, create_dataset, prepare_training_data

DATA_DIR = "../data/RawData"
CHECKPOINT_DIR = "../checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "soh_model_nn.pth")
SCALER_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")

def run_demo():
    print("=== Battery SOH Prediction Demo ===")
    
    # 1. Check dependencies
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or Scaler not found in checkpoints/.")
        print("Please run 'python src/main.py' first to train the model.")
        return

    # 2. Load Model
    print("Loading Model...")
    scaler = joblib.load(SCALER_PATH)
    input_dim = scaler.mean_.shape[0]
    model = SOHNet(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 3. Load Sample Data
    print("Loading Sample Data...")
    # Load just one file for speed if possible, otherwise load all and pick random
    # We'll use load_all_data but maybe limit scope if it accepts it? No, but it's fine.
    # Actually, let's just pick one random file from data dir
    import glob
    files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    if not files:
        print("No .mat files found in data/RawData.")
        return
        
    # Prefer B0005 if present (known good), else random
    b0005 = [f for f in files if "B0005" in f]
    if b0005:
        sample_file = b0005[0]
    else:
        sample_file = files[0]
        
    print(f"Using battery file: {os.path.basename(sample_file)}")
    
    # We need to import load_battery_data specifically if we want just one
    from utils import load_battery_data
    cycles = load_battery_data(sample_file)
    
    if not cycles:
        print("No valid cycles found.")
        return
        
    # Pick a random cycle
    cycle_idx = np.random.randint(0, len(cycles))
    sample_cycle = cycles[cycle_idx]
    
    print(f"Selected Cycle ID: {sample_cycle['cycle_id']}")
    print(f"Actual SOH: {sample_cycle['soh']:.4f}")
    
    # 4. Preprocess
    snippets, _ = create_dataset([sample_cycle], snippets_per_cycle=5, min_duration=10, max_duration=10)
    
    if not snippets:
        print("Could not extract snippets from this cycle.")
        return
        
    # 5. Predict
    df = prepare_training_data(snippets)
    X_scaled = scaler.transform(df)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        preds = model(X_tensor).numpy().flatten()
        
    avg_pred = np.mean(preds)
    
    print(f"\nPredictions on 5 random snippets: {preds}")
    print(f"Average Predicted SOH: {avg_pred:.4f}")
    print(f"Error: {abs(avg_pred - sample_cycle['soh']):.4f}")
    
    print("\nDemo Completed Successfully.")

if __name__ == "__main__":
    run_demo()
