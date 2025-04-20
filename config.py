# Shared config
INPUT_DIM = 12
HIDDEN_DIMS = [16, 64]
OUTPUT_DIM = 3
LEARNING_RATE = 1e-3
EPOCHS = 1000
BATCH_SIZE = 64
MEMORY_SIZE = 2000
GAMMA = 0.9
EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
SYNC_FREQ = 100
MAX_MOVES = 10
MODE = "random" # or 'static'

REWARD_CAUGHT = -1
REWARD_SURVIVED = 1
STEP_PENALTY = 0.01
STATIC_MOUSE_START = 0
STATIC_CAT_START = 4

# Mouse training config
mouse_config = {
    "input_dim": INPUT_DIM,
    "hidden_dims": HIDDEN_DIMS,
    "output_dim": OUTPUT_DIM,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "memory_size": MEMORY_SIZE,
    "gamma": GAMMA,
    "epsilon_start": EPSILON_START,
    "epsilon_end": EPSILON_END,
    "epsilon_decay": EPSILON_DECAY,
    "sync_freq": SYNC_FREQ,
    "max_moves": MAX_MOVES,
    "model_path": "saved_models/mouse_model.pt",
    "role": "mouse",
    "mode": MODE
}

# Cat training config
cat_config = {
    **mouse_config,  # inherit shared settings
    "model_path": "saved_models/cat_model.pt",
    "role": "cat"
}
