# =========================
# MLP Agent Hyperparameters
# =========================


mlp_hidden_dims = (256) # Tuple of hidden layer sizes
mlp_activation = "leaky_relu"  # Activation function: "relu", "tanh", "leaky_relu", "elu", "gelu"
mlp_dropout = 0.1        # Dropout rate for MLP layers (0.0 = no dropout)

def get_lstm_mlp_hidden_dims():
    """Return hidden layer dimensions for MLP agent."""
    d = mlp_hidden_dims
    if isinstance(d, int):
        return (d, )
    return tuple(d)

# Enhanced MLP agent parameters
use_batch_norm = True
use_dueling_architecture = True

# Experience replay parameters
sequence_sampling_strategy = "prioritized"  # Options: uniform, prioritized, recency_biased

# Priority experience replay parameters
prio_alpha = 0.6  # Controls how much prioritization is used (0 = no prioritization, 1 = full prioritization)
prio_beta = 0.4   # Controls importance sampling weight adjustment (should be annealed to 1)
prio_beta_annealing = 0.00001  # How quickly to increase beta to reduce bias
prio_epsilon = 0.01  # Small constant to ensure all transitions have non-zero priority

# Learning rate settings
lr_schedule = [
    (0, 0.0001),        # Initial learning rate
    (100000, 0.001),    # Warmup to higher rate
    (1000000, 0.0001),  # Gradual decay
    (2000000, 0.00001), # Final learning rate
]

# Gradient clipping settings
clip_grad_norm = 10.0
clip_grad_value = 10.0

# Target network update settings
soft_update_tau = 0.005  # Soft update coefficient (1.0 for hard update)
