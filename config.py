# === General Settings ===
NUM_NODES = 6  # Game has 6 noes
MAX_MOVES = 15  # Max moves before cat losses
MODE = "random"  # or 'static', 'random' is harder

STATIC_MOUSE_START = 0
STATIC_CAT_START = 4

# === DQN Architecture ===
INPUT_DIM = NUM_NODES * 2  # one-hot for mouse + one-hot for cat
HIDDEN_DIMS = [64, 128, 64]
OUTPUT_DIM = (
    3  # Max number of neighbor actions from any node (will this crash for some nodes?)
)

# === Training ===
GAMMA = 0.9  # Discount factor
EPSILON_START = 0.3  # epsilon value for epsilon-greedy
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 1e-3  # lr for DQN
BATCH_SIZE = 64
MEMORY_SIZE = 2_000  # for experiance replay
EPOCHS = 1_000
SYNC_FREQ = 100

# === Reward Structure ===
REWARD_CAUGHT = -1.0  # mouse gets this if caught
REWARD_SURVIVED = 1.0
STEP_PENALTY = -0.01

# === Model Checkpoints ===
MOUSE_MODEL_PATH = "saved_models/mouse_model.pt"
CAT_MODEL_PATH = "saved_models/cat_model.pt"
