"""
Optimized LSTM Agent for Trackmania RL

This module implements an optimized LSTM-based agent with:
- Advanced architectural improvements
- Memory-efficient processing
- Better gradient flow
- Optional attention mechanisms
- Improved normalization and regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.typing as npt
from typing import Tuple, Optional, Dict, Any
import math

from config_files import config_copy
from config_files import lstm_config_optimized_copy as lstm_config


def get_activation(name: str):
    """Get activation function by name with additional options."""
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
    elif name == "swish" or name == "silu":
        return nn.SiLU(inplace=True)
    elif name == "mish":
        return nn.Mish(inplace=True)
    else:
        raise ValueError(f"Unsupported activation function: {name}")


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence modeling."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for sequence modeling."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        return output, attn_weights


class OptimizedCNNHead(nn.Module):
    """Optimized CNN head with better feature extraction."""
    
    def __init__(self, img_shape: Tuple[int, int, int], activation: str = "gelu", dropout: float = 0.1):
        super().__init__()
        c, h, w = img_shape
        assert c == 1, "Expected single-channel images"
        
        # More sophisticated CNN architecture
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            get_activation(activation),
            nn.Dropout2d(dropout),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.Dropout2d(dropout),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.Dropout2d(dropout),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation),
            nn.AdaptiveAvgPool2d((4, 4)),  # Ensure consistent output size
            nn.Flatten(),
        )
        
        # Compute output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            self.output_dim = self.conv_layers(dummy).shape[1]
    
    def forward(self, x):
        return self.conv_layers(x)


class OptimizedLSTMAgent(nn.Module):
    """
    Optimized LSTM-based agent with advanced features.
    """
    
    def __init__(
        self,
        float_inputs_dim: int,
        img_shape: Tuple[int, int, int],
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        mlp_hidden_dims: Tuple[int, ...],
        n_actions: int,
        float_inputs_mean: npt.NDArray,
        float_inputs_std: npt.NDArray,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_attention: bool = False,
        use_layer_norm: bool = True,
        use_residual: bool = False,
        use_positional_encoding: bool = False,
    ):
        super().__init__()
        
        # Register normalization parameters
        self.register_buffer("float_inputs_mean", torch.tensor(float_inputs_mean, dtype=torch.float32))
        self.register_buffer("float_inputs_std", torch.tensor(float_inputs_std, dtype=torch.float32))
        
        # Store configuration
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_positional_encoding = use_positional_encoding
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # Optimized CNN head
        self.img_head = OptimizedCNNHead(img_shape, activation, dropout)
        
        # Input dimensions
        self.float_inputs_dim = float_inputs_dim
        self.lstm_input_dim = self.img_head.output_dim + float_inputs_dim
        
        # Input projection layer for better feature mixing
        self.input_projection = nn.Sequential(
            nn.Linear(self.lstm_input_dim, lstm_hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(lstm_hidden_dim)
        
        # LSTM layers with improved configuration
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=False,  # Keep unidirectional for causal modeling
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_hidden_dim)
        
        # Attention mechanism
        if use_attention:
            self.attention = MultiHeadAttention(
                lstm_hidden_dim, 
                n_heads=getattr(lstm_config, 'attention_heads', 8),
                dropout=dropout
            )
            if use_layer_norm:
                self.attn_norm = nn.LayerNorm(lstm_hidden_dim)
        
        # Enhanced MLP head with residual connections
        mlp_layers = []
        input_dim = lstm_hidden_dim
        
        for i, h in enumerate(mlp_hidden_dims):
            mlp_layers.extend([
                nn.Linear(input_dim, h),
                nn.LayerNorm(h) if use_layer_norm else nn.Identity(),
                get_activation(activation),
                nn.Dropout(dropout)
            ])
            input_dim = h
        
        # Output layer
        mlp_layers.append(nn.Linear(input_dim, n_actions))
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with better initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weight initialization
                    nn.init.orthogonal_(param)
                elif 'conv' in name:
                    # CNN weight initialization
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    # Linear layer initialization
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(
        self,
        img_seq: torch.Tensor,
        float_inputs_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass with optimized processing.
        
        Args:
            img_seq: (batch, seq_len, 1, H, W)
            float_inputs_seq: (batch, seq_len, float_inputs_dim)
            hidden: LSTM hidden state tuple
            return_attention: Whether to return attention weights
            
        Returns:
            Q-values: (batch, seq_len, n_actions)
            hidden: Updated LSTM hidden state
            attention_weights: Attention weights if requested
        """
        batch_size, seq_len, c, h, w = img_seq.shape
        
        # Process images through CNN
        img_flat = img_seq.view(batch_size * seq_len, c, h, w)
        img_feat = self.img_head(img_flat)
        img_feat = img_feat.view(batch_size, seq_len, -1)
        
        # Normalize float inputs
        float_inputs_norm = (float_inputs_seq - self.float_inputs_mean) / (self.float_inputs_std + 1e-8)
        
        # Handle dimension matching
        if float_inputs_norm.dim() == 2:
            float_inputs_norm = float_inputs_norm.unsqueeze(1).expand(-1, seq_len, -1)
        elif float_inputs_norm.dim() == 3:
            if float_inputs_norm.shape[1] != seq_len:
                raise ValueError(f"Sequence length mismatch: {float_inputs_norm.shape[1]} vs {seq_len}")
        
        # Concatenate and project features
        features = torch.cat([img_feat, float_inputs_norm], dim=-1)
        features = self.input_projection(features)
        
        # Add positional encoding
        if self.use_positional_encoding:
            features = self.pos_encoding(features.transpose(0, 1)).transpose(0, 1)
        
        # LSTM processing
        lstm_out, new_hidden = self.lstm(features, hidden)
        
        # Apply layer normalization
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
        
        # Apply attention mechanism
        attention_weights = None
        if self.use_attention:
            attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Residual connection
            if self.use_residual:
                lstm_out = lstm_out + attn_out
            else:
                lstm_out = attn_out
            
            # Layer normalization after attention
            if self.use_layer_norm:
                lstm_out = self.attn_norm(lstm_out)
        
        # MLP head
        q_values = self.mlp(lstm_out)
        
        if return_attention:
            return q_values, new_hidden, attention_weights
        else:
            return q_values, new_hidden
    
    def get_feature_representations(self, img_seq, float_inputs_seq):
        """Get intermediate feature representations for analysis."""
        batch_size, seq_len, c, h, w = img_seq.shape
        
        # CNN features
        img_flat = img_seq.view(batch_size * seq_len, c, h, w)
        img_feat = self.img_head(img_flat)
        img_feat = img_feat.view(batch_size, seq_len, -1)
        
        # Normalized float features
        float_inputs_norm = (float_inputs_seq - self.float_inputs_mean) / (self.float_inputs_std + 1e-8)
        if float_inputs_norm.dim() == 2:
            float_inputs_norm = float_inputs_norm.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combined features
        features = torch.cat([img_feat, float_inputs_norm], dim=-1)
        projected_features = self.input_projection(features)
        
        # LSTM features
        lstm_out, _ = self.lstm(projected_features)
        
        return {
            'cnn_features': img_feat,
            'float_features': float_inputs_norm,
            'projected_features': projected_features,
            'lstm_features': lstm_out
        }


def make_optimized_lstm_agent(
    jit: bool = False,
    is_inference: bool = False,
    use_attention: bool = False,
    use_layer_norm: bool = True,
    use_residual: bool = False,
    use_positional_encoding: bool = False,
):
    """
    Factory function to create an optimized LSTM agent.
    """
    
    # Get configuration parameters
    activation = getattr(lstm_config, 'lstm_activation', 'gelu')
    dropout = getattr(lstm_config, 'lstm_dropout', 0.1)
    
    # Override with config values if available
    use_attention = getattr(lstm_config, 'use_attention_mechanism', use_attention)
    use_layer_norm = getattr(lstm_config, 'use_layer_normalization', use_layer_norm)
    use_residual = getattr(lstm_config, 'use_residual_connections', use_residual)
    use_positional_encoding = getattr(lstm_config, 'use_positional_encoding', use_positional_encoding)
    
    agent = OptimizedLSTMAgent(
        float_inputs_dim=config_copy.float_input_dim,
        img_shape=(1, config_copy.H_downsized, config_copy.W_downsized),
        lstm_hidden_dim=lstm_config.lstm_hidden_dim,
        lstm_num_layers=lstm_config.lstm_num_layers,
        mlp_hidden_dims=lstm_config.get_lstm_mlp_hidden_dims(),
        n_actions=len(config_copy.inputs),
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
        activation=activation,
        dropout=dropout,
        use_attention=use_attention,
        use_layer_norm=use_layer_norm,
        use_residual=use_residual,
        use_positional_encoding=use_positional_encoding,
    )
    
    # Apply compilation/JIT if requested
    if jit:
        if config_copy.is_linux:
            compile_mode = None if "rocm" in torch.__version__ else (
                "max-autotune" if is_inference else "max-autotune-no-cudagraphs"
            )
            model = torch.compile(agent, dynamic=False, mode=compile_mode)
        else:
            # Note: JIT scripting might not work with all optimizations
            try:
                model = torch.jit.script(agent)
            except Exception as e:
                print(f"JIT scripting failed: {e}, using regular model")
                model = agent
    else:
        import copy
        model = copy.deepcopy(agent)
    
    return (
        model.to(device="cuda"),
        agent.to(device="cuda"),
    )


# Backward compatibility
def make_untrained_lstm_agent(jit: bool, is_inference: bool):
    """Backward compatibility wrapper."""
    return make_optimized_lstm_agent(jit, is_inference)