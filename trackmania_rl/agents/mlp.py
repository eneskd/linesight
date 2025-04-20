
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple

from config_files import config_copy
from config_files import mlp_config_copy


def get_activation(name: str):
    """Return the activation nn.Module corresponding to the given name."""
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


class MLPAgent(nn.Module):
    """
    An enhanced MLP agent for DQN-style RL, supporting both single-channel image (frame) and float inputs.
    Features improved image processing, batch normalization, and dueling architecture.
    Expects image input shape to be (batch_size, 1, H, W).
    """
    def __init__(
        self,
        float_inputs_dim: int,
        img_shape: Tuple[int, int, int],  # (C, H, W)
        hidden_dims: Tuple[int, ...],
        n_actions: int,
        float_inputs_mean: npt.NDArray,
        float_inputs_std: npt.NDArray,
        activation: str = "relu",
        dropout: float = 0.0,
        use_norm: bool = True,  # Renamed from use_batch_norm
        use_dueling: bool = True,
    ):
        super().__init__()
        # Register mean and std as buffers so they move with the model's device
        self.register_buffer("float_inputs_mean", torch.tensor(float_inputs_mean, dtype=torch.float32))
        self.register_buffer("float_inputs_std", torch.tensor(float_inputs_std, dtype=torch.float32))
        self.use_dueling = use_dueling

        # Expect single-channel images: C = 1
        c, h, w = img_shape
        assert c == 1, "MLPAgent expects single-channel images with shape (1, H, W)"
        
        # Enhanced image processing with more channels and group normalization
        # (GroupNorm with num_groups=1 is equivalent to LayerNorm for conv layers)
        layers = []
        layers.append(nn.Conv2d(1, 16, kernel_size=5, stride=2))
        if use_norm:
            # Use GroupNorm instead of BatchNorm2d
            layers.append(nn.GroupNorm(4, 16))  # 4 groups for 16 channels
        layers.append(get_activation(activation))
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Conv2d(16, 32, kernel_size=3, stride=2))
        if use_norm:
            layers.append(nn.GroupNorm(8, 32))  # 8 groups for 32 channels
        layers.append(get_activation(activation))
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=1))
        if use_norm:
            layers.append(nn.GroupNorm(16, 64))  # 16 groups for 64 channels
        layers.append(get_activation(activation))
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Flatten())
        self.img_head = nn.Sequential(*layers)
        
        # Compute flattened conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.img_head(dummy).shape[1]

        # MLP for combined features
        input_dim = float_inputs_dim + conv_out_dim
        mlp_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, h))
            if use_norm:
                # Use LayerNorm instead of BatchNorm1d - works with any batch size
                mlp_layers.append(nn.LayerNorm(h))
            mlp_layers.append(get_activation(activation))
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = h
        
        self.feature_layers = nn.Sequential(*mlp_layers)
        
        # Always create all layers for TorchScript compatibility
        # but only use the appropriate ones based on self.use_dueling
        self.advantage = nn.Linear(prev_dim, n_actions)
        self.value = nn.Linear(prev_dim, 1)
        self.output_layer = nn.Linear(prev_dim, n_actions)

    def forward(self, img: torch.Tensor, float_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (batch_size, 1, H, W)
            float_inputs: (batch_size, float_inputs_dim)
        Returns:
            Q-values: (batch_size, n_actions)
        """
        # Normalize float inputs with epsilon for stability
        x_float = (float_inputs - self.float_inputs_mean) / (self.float_inputs_std + 1e-8)
        
        # Process image
        x_img = self.img_head(img)
        
        # Concatenate and process through feature layers
        x = torch.cat([x_float, x_img], dim=1)
        features = self.feature_layers(x)
        
        # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        if self.use_dueling:
            advantages = self.advantage(features)
            values = self.value(features)
            return values + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            return self.output_layer(features)


def make_untrained_mlp_agent(jit: bool, is_inference: bool) -> Tuple[MLPAgent, MLPAgent]:
    """
    Factory function to create two identical MLP agents (compiled and uncompiled).
    """
    hidden_dims = mlp_config_copy.get_lstm_mlp_hidden_dims()
    # Use single-channel image shape: (1, H, W)
    img_shape = (1, config_copy.H_downsized, config_copy.W_downsized)
    activation = getattr(mlp_config_copy, "mlp_activation", "relu")
    dropout = getattr(mlp_config_copy, "mlp_dropout", 0.0)
    use_norm = getattr(mlp_config_copy, "use_batch_norm", True)  # Keep the config name for backward compatibility
    use_dueling = getattr(mlp_config_copy, "use_dueling", True)
    
    uncompiled_model = MLPAgent(
        float_inputs_dim=config_copy.float_input_dim,
        img_shape=img_shape,
        hidden_dims=hidden_dims,
        n_actions=len(config_copy.inputs),
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
        activation=activation,
        dropout=dropout,
        use_norm=use_norm,
        use_dueling=use_dueling,
    )
    if jit:
        if config_copy.is_linux:
            compile_mode = None if "rocm" in torch.__version__ else ("max-autotune" if is_inference else "max-autotune-no-cudagraphs")
            model = torch.compile(uncompiled_model, dynamic=False, mode=compile_mode)
        else:
            model = torch.jit.script(uncompiled_model)
    else:
        import copy
        model = copy.deepcopy(uncompiled_model)
    return (
        model.to(device="cuda"),  # compiled or scripted model
        uncompiled_model.to(device="cuda"),  # uncompiled model for weight sharing
    )
