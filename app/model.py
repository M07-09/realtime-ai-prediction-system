"""
The LSTM network used for the forecast.

Architecture
------------
    Input   (batch, 60 time-steps, 7 features)
      |
    LSTM    2 stacked layers, 96 hidden units, dropout 0.2 between layers
      |
    take the hidden state of the LAST time-step
      |
    Dropout -> Linear(96 -> 32) -> ReLU -> Linear(32 -> 1)
      |
    Output  (batch, 1)  = predicted standardised log return of the next bar

The predicted price is reconstructed by the predictor as

    price_hat = last_close * exp(predicted_log_return)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from app.config import DROPOUT, HIDDEN_SIZE, NUM_LAYERS


class PriceLSTM(nn.Module):
    """Sequence-to-one LSTM regressor."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]            # (batch, hidden_size)
        return self.head(self.dropout(last_step)).squeeze(-1)

    # ------------------------------------------------------------ helpers
    def config(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }


def save_model(model: PriceLSTM, path: Path, extra: Dict[str, Any] | None = None) -> None:
    """Persist weights together with the architecture description."""
    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": model.config(),
    }
    if extra:
        payload["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model(path: Path, device: torch.device | str = "cpu") -> PriceLSTM:
    """Rebuild the network from a checkpoint saved by save_model()."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = PriceLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def pick_device(preference: str = "auto") -> torch.device:
    """Choose CUDA when it is available, otherwise CPU."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")
