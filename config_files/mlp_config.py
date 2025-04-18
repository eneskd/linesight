# =========================
# MLP Agent Hyperparameters
# =========================


mlp_hidden_dims = (256) # Tuple of hidden layer sizes
mlp_activation = "relu"  # Activation function: "relu", "tanh", "leaky_relu", "elu", "gelu"
mlp_dropout = 0.00        # Dropout rate for MLP layers (0.0 = no dropout)

def get_lstm_mlp_hidden_dims():
    d = mlp_hidden_dims
    if isinstance(d, int):
        return (d, )
    return tuple(d)