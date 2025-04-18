
import torch
import torch.nn as nn
import numpy.typing as npt
from typing import Tuple, Optional  # <-- Add Optional

from config_files import config_copy
from config_files import lstm_config_copy

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class LSTMAgent(nn.Module):
    """
    LSTM-based agent for DQN-style RL.
    Processes sequences of (image, float) inputs and outputs Q-values for each action at each time step.
    """
    def __init__(
        self,
        float_inputs_dim: int,
        img_shape: Tuple[int, int, int],  # (C, H, W)
        cnn_out_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        mlp_hidden_dims: Tuple[int, ...],
        n_actions: int,
        float_inputs_mean: npt.NDArray,
        float_inputs_std: npt.NDArray,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        # Register mean and std as buffers so they move with the model's device
        self.register_buffer("float_inputs_mean", torch.tensor(float_inputs_mean, dtype=torch.float32))
        self.register_buffer("float_inputs_std", torch.tensor(float_inputs_std, dtype=torch.float32))

        c, h, w = img_shape
        assert c == 1, "LSTMAgent expects single-channel images with shape (1, H, W)"
        self.img_head = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        # Compute flattened conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.img_head(dummy).shape[1]

        self.cnn_out_dim = conv_out_dim
        self.float_inputs_dim = float_inputs_dim
        self.lstm_input_dim = conv_out_dim + float_inputs_dim

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # MLP head after LSTM
        mlp_layers = []
        input_dim = lstm_hidden_dim
        for h in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, h))
            mlp_layers.append(get_activation(activation))
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = h
        mlp_layers.append(nn.Linear(input_dim, n_actions))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        img_seq: torch.Tensor,
        float_inputs_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # <-- Add explicit type
    ):
        """
        Args:
            img_seq: (batch, seq_len, 1, H, W)
            float_inputs_seq: (batch, seq_len, float_inputs_dim)
            hidden: (h_0, c_0) tuple for LSTM initial state (optional)
        Returns:
            Q-values: (batch, seq_len, n_actions)
            hidden: (h_n, c_n) tuple for LSTM final state
        """
        batch, seq_len, c, h, w = img_seq.shape
        # Flatten batch and seq for CNN
        img_flat = img_seq.view(batch * seq_len, c, h, w)
        img_feat = self.img_head(img_flat)  # (batch*seq_len, cnn_out_dim)
        img_feat = img_feat.view(batch, seq_len, -1)  # (batch, seq_len, cnn_out_dim)

        # Normalize float inputs
        float_inputs_norm = (float_inputs_seq - self.float_inputs_mean) / (self.float_inputs_std + 1e-8)

        # Ensure dims match for concatenation: both [batch, seq_len, ...]
        if float_inputs_norm.dim() == 2:
            # Expand float_inputs_norm across sequence length
            float_inputs_norm = float_inputs_norm.unsqueeze(1).expand(-1, seq_len, -1)
        elif float_inputs_norm.dim() == 3:
            if float_inputs_norm.shape[1] != seq_len:
                raise ValueError(f"float_inputs_seq dims do not match sequence length: got {float_inputs_norm.shape[1]}, expected {seq_len}")
        else:
            raise ValueError(f"float_inputs_seq must have 2 or 3 dims (got shape {float_inputs_norm.shape})")

        # Concatenate features
        features = torch.cat([img_feat, float_inputs_norm], dim=-1)  # (batch, seq_len, lstm_input_dim)

        # LSTM
        lstm_out, hidden = self.lstm(features, hidden)  # (batch, seq_len, lstm_hidden_dim)

        # MLP head (apply to each time step)
        q_values = self.mlp(lstm_out)  # (batch, seq_len, n_actions)
        return q_values, hidden

def make_untrained_lstm_agent(
    jit: bool,
    is_inference: bool,
):
    """
    Factory function to create an LSTM agent (optionally JIT-compiled).
    """

    agent = LSTMAgent(
        float_inputs_dim=config_copy.float_input_dim,
        img_shape=(1, config_copy.H_downsized, config_copy.W_downsized),
        cnn_out_dim=lstm_config_copy.lstm_cnn_out_dim,
        lstm_hidden_dim=lstm_config_copy.lstm_hidden_dim,
        lstm_num_layers=lstm_config_copy.lstm_num_layers,
        mlp_hidden_dims=lstm_config_copy.get_lstm_mlp_hidden_dims(),
        n_actions=len(config_copy.inputs),
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
        activation=lstm_config_copy.lstm_activation,
        dropout=lstm_config_copy.lstm_dropout,
    )
    if jit:
        if config_copy.is_linux:
            compile_mode = None if "rocm" in torch.__version__ else ("max-autotune" if is_inference else "max-autotune-no-cudagraphs")
            model = torch.compile(agent, dynamic=False, mode=compile_mode)
        else:
            model = torch.jit.script(agent)
    else:
        import copy
        model = copy.deepcopy(agent)
    return (
        model.to(device="cuda"),
        agent.to(device="cuda"),
    )
