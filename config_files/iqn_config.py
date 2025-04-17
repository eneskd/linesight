import numpy as np
from config_files import config_copy

# IQN network architecture and algorithm parameters
float_input_dim = 27 + 3 * config_copy.n_zone_centers_in_inputs + 4 * config_copy.n_prev_actions_in_inputs + 4 * config_copy.n_contact_material_physics_behavior_types + 1
float_hidden_dim = 256
conv_head_output_dim = 5632
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64
iqn_n = 8  # must be an even number because we sample tau symmetrically around 0.5
iqn_k = 32  # must be an even number because we sample tau symmetrically around 0.5
iqn_kappa = 5e-3
use_ddqn = False

