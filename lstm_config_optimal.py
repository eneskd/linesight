# Optimal LSTM Configuration for Trackmania RL
# Tuned for performance, memory efficiency, and training stability

# ========================================
# Core Architecture Parameters
# ========================================

# LSTM hidden dimension - balanced for capacity vs memory
# 512 provides good representational power without excessive memory usage
lstm_hidden_dim = 512

# Single layer LSTM for faster training and less overfitting
# Multiple layers often don't help much in RL and increase training time
lstm_num_layers = 1

# MLP head - two-layer design for good function approximation
# Gradually decreasing sizes help with feature refinement
lstm_mlp_hidden_dims = (512, 256)

# GELU activation - better gradient flow than ReLU for deep networks
lstm_activation = "gelu"

# Moderate dropout for regularization without hurting learning
lstm_dropout = 0.1

# ========================================
# Training Batch and Sequence Parameters
# ========================================

# Batch size - larger than original for better gradient estimates
# 32 is good balance between memory usage and training stability
lstm_batch_size = 32

# Sequence length - longer sequences for better temporal modeling
# 16 captures more temporal dependencies than 8, but not too memory intensive
lstm_seq_len = 16

# Large replay buffer for diverse experience
# 200k provides good sample diversity without excessive memory usage
replay_buffer_capacity = 200000

# ========================================
# Optimizer and Learning Parameters
# ========================================

# AdamW optimizer - better weight decay handling than RAdam
optimizer = "AdamW"

# Moderate weight decay for regularization
weight_decay = 0.01

# No gradient accumulation by default - use larger batch size instead
gradient_accumulation_steps = 1

# Conservative gradient clipping for stability
max_grad_norm = 0.5
max_grad_value = 0.1

# ========================================
# Loss Function and Training Features
# ========================================

# Huber loss for robustness to outliers
use_huber_loss = True
huber_delta = 1.0

# Discount factor - high for long-term planning in racing
gamma = 0.995

# Temporal weighting - emphasize recent timesteps
temporal_weighting = True

# Prioritized replay for better sample efficiency
use_prioritized_replay = True

# ========================================
# Network Update Frequencies
# ========================================

# More frequent target updates for faster learning
target_update_frequency = 500

# Moderate training steps per iteration to prevent blocking
max_training_steps_per_iteration = 15

# ========================================
# Logging and Monitoring
# ========================================

# Frequent logging for monitoring
log_frequency = 1
tensorboard_log_frequency = 5

# ========================================
# Advanced Architecture Features
# ========================================

# Layer normalization for training stability
use_layer_normalization = True

# No attention mechanism initially - can be enabled later
use_attention_mechanism = False
use_residual_connections = False
use_positional_encoding = False

# Attention parameters (if enabled later)
attention_heads = 8

# ========================================
# Memory and Performance Optimizations
# ========================================

# Enable mixed precision for faster training (if stable)
use_mixed_precision = False  # Start disabled, enable if training is stable

# Memory optimization flags
pin_memory = True
non_blocking_transfer = True

# ========================================
# Helper Functions
# ========================================

def get_lstm_mlp_hidden_dims():
    """Get MLP hidden dimensions as tuple."""
    if isinstance(lstm_mlp_hidden_dims, int):
        return (lstm_mlp_hidden_dims,)
    return tuple(lstm_mlp_hidden_dims)

# ========================================
# Validation and Info
# ========================================

print("=== Optimal LSTM Configuration Loaded ===")
print(f"Architecture: {lstm_hidden_dim}D LSTM → {get_lstm_mlp_hidden_dims()} MLP")
print(f"Training: batch={lstm_batch_size}, seq_len={lstm_seq_len}")
print(f"Memory: buffer={replay_buffer_capacity:,}, prioritized={use_prioritized_replay}")
print(f"Optimizer: {optimizer} (wd={weight_decay})")
print(f"Features: Huber loss, temporal weighting, layer norm")
print("==========================================")

# ========================================
# Parameter Justifications
# ========================================

"""
PARAMETER JUSTIFICATIONS:

1. lstm_hidden_dim = 512
   - Provides good representational capacity
   - Not too large to cause memory issues
   - Sweet spot for most RL tasks

2. lstm_seq_len = 16
   - Captures ~800ms of history (16 * 50ms per action)
   - Good for racing decisions (cornering, overtaking)
   - Longer than original 8 for better temporal modeling

3. lstm_batch_size = 32
   - Larger than original 16 for more stable gradients
   - Still manageable memory-wise with seq_len=16
   - Good balance for GPU utilization

4. replay_buffer_capacity = 200000
   - 4x larger than original for better diversity
   - Stores ~2.7M individual transitions (200k * 16 seq_len)
   - Provides good coverage of state space

5. gamma = 0.995
   - Higher than standard 0.99 for long-term planning
   - Important for racing where early actions affect lap time
   - Matches the temporal nature of racing

6. target_update_frequency = 500
   - More frequent than original 1000 for faster learning
   - Balances stability vs adaptation speed
   - Good for dynamic racing environments

7. use_prioritized_replay = True
   - Improves sample efficiency significantly
   - Focuses on important transitions (crashes, good moves)
   - Essential for complex environments like racing

8. temporal_weighting = True
   - Recent actions more important in racing
   - Helps with credit assignment
   - Improves learning of immediate consequences

9. optimizer = "AdamW"
   - Better weight decay handling than RAdam
   - More stable training in practice
   - Good default for transformer-style architectures

10. use_layer_normalization = True
    - Stabilizes training with longer sequences
    - Helps with gradient flow in deeper networks
    - Reduces internal covariate shift
"""