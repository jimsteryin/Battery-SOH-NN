import torch
import torch.nn as nn

class SOHNet(nn.Module):
    """
    Multi-Layer Perceptron for SOH prediction using statistical features.
    """
    def __init__(self, input_dim):
        super(SOHNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SOHGRU(nn.Module):
    """
    Gated Recurrent Unit for SOH prediction using raw sequences.
    """
    def __init__(self, input_dim, hidden_dim=96):
        super(SOHGRU, self).__init__()
        # GRU: layers=1, hidden=96
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        # Linear(64) - Note: Input is hidden_dim (96)
        self.fc1 = nn.Linear(hidden_dim, 64)
        # ReLU
        self.relu = nn.ReLU()
        # Linear(1)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (Batch, Seq, Features)
        out, _ = self.gru(x)
        
        # Take the last time step output
        # out shape: (Batch, Seq, Hidden) -> Last step: (Batch, Hidden)
        out = out[:, -1, :]
        
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
