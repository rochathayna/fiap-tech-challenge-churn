import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        logger.info(f"ChurnMLP criado com input_dim={input_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)