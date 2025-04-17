
from config_files import config_copy

# =========================
# MLP Agent Hyperparameters
# =========================


float_input_dim = 27 + 3 * config_copy.n_zone_centers_in_inputs + 4 * config_copy.n_prev_actions_in_inputs + 4 * config_copy.n_contact_material_physics_behavior_types + 1
mlp_hidden_dims = (256, 256, 256)  # Tuple of hidden layer sizes
mlp_activation = "relu"  # Activation function: "relu", "tanh", "leaky_relu", "elu", "gelu"
mlp_dropout = 0.05        # Dropout rate for MLP layers (0.0 = no dropout)