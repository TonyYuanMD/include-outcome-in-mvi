# gan_models.py
import torch.nn as nn
import torch

class Generator(nn.Module):
    """
    Generator network for GAIN-like GAN.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        total_input_dim = input_dim + input_dim  # x (input_dim) + m (input_dim for mask)
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, input_dim)  # Output matches input_dim (imputed values)
    
    def forward(self, x, m):
        xm = torch.cat([x, m], dim=1)
        x = torch.relu(self.fc1(xm))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Linear output for imputed values

class Discriminator(nn.Module):
    """
    Discriminator network for GAIN-like GAN.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        total_input_dim = input_dim + input_dim
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Output a single value for binary classification
    
    def forward(self, x, m, g_out=None):
        # Extend m to match x's feature dimension by repeating for non-missing columns
        m_extended = m.expand(-1, x.shape[1])  # Repeat m to match x's feature count
        if g_out is not None:
            imputed = x * (1 - m_extended) + g_out * m_extended  # Apply mask to all features
        else:
            imputed = x
        xm = torch.cat([imputed, m], dim=1)  # Concatenate with original mask
        x = torch.relu(self.fc1(xm))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Probability of being real

# Documentation
"""
Classes:
- Generator: Feedforward NN for imputing missing values using configurable hidden_dim, taking x and m as input.
- Discriminator: Feedforward NN for distinguishing real vs. imputed values with configurable hidden_dim.
"""