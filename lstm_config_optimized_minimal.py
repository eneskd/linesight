# Essential LSTM Configuration Variables
# Based on actual usage in the codebase

# ========================================
# Core Architecture Parameters (Required)
# ========================================

# LSTM architecture
lstm_hidden_dim = 512
lstm_num_layers = 1
lstm_mlp_hidden_dims = (512, 256)  # Tuple of hidden layer sizes
lstm_activation = "gelu"
lstm_dropout = 0.1

# ========================================
# Training Parameters (Required)
# ========================================

# Batch and sequence settings
lstm_batch_size = 32
lstm_seq_len = 16
replay_buffer_capacity = 200000

# Optimizer settings
optimizer = "AdamW"  # Used in learner_process_lstm_optimized.py
weight_decay = 0.01

# Gradient settings
gradient_accumulation_steps = 1  # Used in OptimizedLSTMTrainer
max_grad_norm = 0.5
max_grad_value = 0.1

# ========================================
# Loss and Training Features (Required)
# ========================================

# Loss function
use_huber_loss = True
huber_delta = 1.0
gamma = 0.99

# Advanced features
temporal_weighting = True
use_prioritized_replay = False

# ========================================
# Training Loop Settings (Required)
# ========================================

# Update frequencies
target_update_frequency = 1000
max_training_steps_per_iteration = 10

# Logging
log_frequency = 1
tensorboard_log_frequency = 10

# ========================================
# Advanced Architecture Features (Optional)
# ========================================

# These are used in the LSTM agent but have defaults
use_attention_mechanism = False
use_layer_normalization = True
use_residual_connections = False
use_positional_encoding = False
attention_heads = 8

# ========================================
# Helper Functions (Required)
# ========================================

def get_lstm_mlp_hidden_dims():
    """Get MLP hidden dimensions as tuple."""
    if isinstance(lstm_mlp_hidden_dims, int):
        return (lstm_mlp_hidden_dims,)
    return tuple(lstm_mlp_hidden_dims)

# ========================================
# Validation
# ========================================

# Ensure all required variables are defined
_required_vars = [
    'lstm_hidden_dim', 'lstm_num_layers', 'lstm_mlp_hidden_dims',
    'lstm_batch_size', 'lstm_seq_len', 'replay_buffer_capacity',
    'gradient_accumulation_steps', 'max_grad_norm', 'max_grad_value',
    'use_huber_loss', 'huber_delta', 'gamma', 'temporal_weighting',
    'target_update_frequency', 'max_training_steps_per_iteration',
    'log_frequency', 'tensorboard_log_frequency'
]

for var in _required_vars:
    if var not in globals():
        raise ValueError(f"Required variable '{var}' not defined in lstm_config_optimized.py")

print("LSTM config loaded successfully!")
print(f"Architecture: {lstm_hidden_dim}D LSTM, {lstm_num_layers} layers, MLP: {get_lstm_mlp_hidden_dims()}")
print(f"Training: batch={lstm_batch_size}, seq_len={lstm_seq_len}, buffer={replay_buffer_capacity}")
print(f"Features: Huber loss={use_huber_loss}, Temporal weighting={temporal_weighting}")