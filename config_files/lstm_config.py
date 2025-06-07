# LSTM Agent Hyperparameters and Architecture

# Import required for accessing general config
from config_files import config_copy

# CNN head output dimension (computed automatically in agent, but you can set for reference)
lstm_cnn_out_dim = 256

# LSTM hidden state dimension
lstm_hidden_dim = 256

# Number of LSTM layers
lstm_num_layers = 1

# MLP head hidden layer sizes after LSTM
lstm_mlp_hidden_dims = (256)

# Activation function for all layers
lstm_activation = "relu"

# Dropout rate for CNN, LSTM, and MLP layers
lstm_dropout = 0.0
lstm_batch_size = 32

replay_buffer_capacity = 50000
lstm_seq_len = 16

gamma = 0.99

# Target network update frequency (in training steps)
target_update_frequency = 1000  # Update target network every 1000 steps

# Optionally, you can add more LSTM-specific settings here as needed

def get_lstm_mlp_hidden_dims():
    d = lstm_mlp_hidden_dims
    if isinstance(d, int):
        return (d, )
    return tuple(d)
