# Optimized LSTM Agent Configuration
# This file contains optimized hyperparameters and settings for improved LSTM training

# Import required for accessing general config
from config_files import config_copy

# ========================================
# Core Architecture Parameters
# ========================================

# CNN head output dimension (computed automatically)
lstm_cnn_out_dim = 256

# LSTM hidden state dimension - optimized for memory/performance balance
lstm_hidden_dim = 512  # Increased for better capacity

# Number of LSTM layers - single layer for speed, can increase for capacity
lstm_num_layers = 1

# MLP head hidden layer sizes after LSTM - deeper network for better representation
lstm_mlp_hidden_dims = (512, 256)  # Two-layer MLP

# Activation function - GELU often works better than ReLU for transformers/LSTMs
lstm_activation = "gelu"

# Dropout rate - moderate dropout for regularization
lstm_dropout = 0.01

gamma = 0.995

# ========================================
# Training Optimization Parameters
# ========================================

# Batch size - larger batches for better gradient estimates
lstm_batch_size = 32

# Sequence length - longer sequences for better temporal modeling
lstm_seq_len = 16

# Gradient accumulation steps - simulate larger batch sizes
gradient_accumulation_steps = 1

# Optimizer choice - AdamW often works better than RAdam for transformers
optimizer = "AdamW"
weight_decay = 0.01

# Learning rate schedule parameters
initial_learning_rate = 1e-4
lr_warmup_steps = 1000
lr_decay_factor = 0.95

# ========================================
# Advanced Training Features
# ========================================

# Loss function improvements
use_huber_loss = True
huber_delta = 1.0

# Temporal weighting - give more importance to recent timesteps
temporal_weighting = True
temporal_decay_factor = 0.95

# Sequence masking - handle variable length sequences properly
use_sequence_masking = True

# Mixed precision training
use_mixed_precision = False

# ========================================
# Memory and Buffer Optimization
# ========================================

# Replay buffer capacity - larger buffer for better sample diversity
replay_buffer_capacity = 200000

# Prioritized experience replay
use_prioritized_replay = True
priority_alpha = 0.6
priority_beta = 0.4
priority_beta_annealing = 0.001

# Sequence sampling strategy
sequence_sampling_strategy = "recency_biased"  # Options: "uniform", "recency_biased", "prioritized"

# ========================================
# Network Update Parameters
# ========================================

# Target network update frequency (in training steps)
target_update_frequency = 500  # More frequent updates for faster learning

# Soft update tau for target network (if using soft updates)
target_update_tau = 0.005

# ========================================
# Training Loop Optimization
# ========================================

# Maximum training steps per iteration to prevent blocking
max_training_steps_per_iteration = 15

# Checkpoint and logging frequency
save_frequency_s = 300  # 5 minutes
tensorboard_log_frequency = 10  # Every 10 steps

# ========================================
# Advanced Regularization
# ========================================

# Gradient clipping parameters
max_grad_norm = 0.5
max_grad_value = 0.1

# LSTM-specific regularization
lstm_weight_decay_multiplier = 0.5  # Reduce weight decay for LSTM layers

# ========================================
# Performance Monitoring
# ========================================

# Enable performance profiling
enable_profiling = False
profile_steps = 100

# Memory optimization settings
pin_memory = True
non_blocking_transfer = True

# ========================================
# Curriculum Learning (Optional)
# ========================================

# Progressive sequence length training
use_curriculum_learning = False
curriculum_start_seq_len = 8
curriculum_end_seq_len = 32
curriculum_steps = 50000

# ========================================
# Auxiliary Tasks (Optional)
# ========================================

# Auxiliary prediction tasks for better representation learning
use_auxiliary_tasks = False
predict_next_state = False
predict_reward = False
auxiliary_loss_weight = 0.1

log_frequency = 1
tensorboard_log_frequency = 5

# ========================================
# Hyperparameter Schedules
# ========================================

def get_lstm_mlp_hidden_dims():
    """Get MLP hidden dimensions as tuple."""
    d = lstm_mlp_hidden_dims
    if isinstance(d, int):
        return (d,)
    return tuple(d)

def get_learning_rate_schedule():
    """Get learning rate schedule for LSTM training."""
    return [
        (0, initial_learning_rate * 0.1),  # Start with lower LR
        (lr_warmup_steps, initial_learning_rate),  # Warmup to full LR
        (50000, initial_learning_rate * lr_decay_factor),  # Decay
        (100000, initial_learning_rate * lr_decay_factor ** 2),
        (200000, initial_learning_rate * lr_decay_factor ** 3),
    ]

def get_sequence_length_schedule():
    """Get curriculum learning schedule for sequence length."""
    if not use_curriculum_learning:
        return [(0, lstm_seq_len)]
    
    return [
        (0, curriculum_start_seq_len),
        (curriculum_steps // 4, curriculum_start_seq_len + 4),
        (curriculum_steps // 2, curriculum_start_seq_len + 8),
        (3 * curriculum_steps // 4, curriculum_start_seq_len + 16),
        (curriculum_steps, curriculum_end_seq_len),
    ]

# ========================================
# Model Architecture Variants
# ========================================

# Alternative architectures to try
use_attention_mechanism = False  # Add attention layers
use_residual_connections = False  # Add skip connections
use_layer_normalization = True   # Add layer norm for stability

# Attention parameters (if enabled)
attention_heads = 8
attention_dim = 64

# ========================================
# Debugging and Analysis
# ========================================

# Enable detailed logging
verbose_logging = False
log_gradient_norms = True
log_activation_stats = False

# Save intermediate representations
save_hidden_states = False
save_attention_weights = False

# ========================================
# Environment-Specific Optimizations
# ========================================

# Trackmania-specific optimizations
use_speed_prediction = False  # Predict car speed as auxiliary task
use_position_encoding = False  # Add positional encoding to sequences
use_map_embedding = False     # Embed map information

# Reward shaping for LSTM
temporal_reward_smoothing = True
reward_smoothing_factor = 0.9

# ========================================
# Experimental Features
# ========================================

# Transformer-style improvements
use_positional_encoding = False
use_multi_head_attention = False
use_feed_forward_network = False

# Memory-efficient training
use_gradient_checkpointing = False
use_activation_checkpointing = False

# Dynamic batching
use_dynamic_batching = False
max_tokens_per_batch = 2048

# print("Optimized LSTM configuration loaded successfully!")
# print(f"Batch size: {lstm_batch_size}, Sequence length: {lstm_seq_len}")
# print(f"Hidden dim: {lstm_hidden_dim}, MLP dims: {get_lstm_mlp_hidden_dims()}")
# print(f"Optimizer: {optimizer}, Learning rate: {initial_learning_rate}")
# print(f"Advanced features: Huber loss={use_huber_loss}, Temporal weighting={temporal_weighting}")
# print(f"Prioritized replay: {use_prioritized_replay}, Curriculum learning: {use_curriculum_learning}")