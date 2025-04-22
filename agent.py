import numpy as np
import torch
import torch.nn as nn
from dqn import DQN
import numpy.typing as npt

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class DQNAgent:

    def __init__(self, input_dim, hidden_dims, output_dim, epsilon=0.15) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_dim, hidden_dims, output_dim).to(self.device)
        self.epsilon = epsilon

    def act(self, state, neighbors, eval=False):
        """
        Selects an action using epsilon-greedy policy over current node's neighbors.
        'neighbors' is a list of reachable nodes from current position.
        """
        action_space = {i: n for i, n in enumerate(neighbors)}  # local idx -> node

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        if not eval and np.random.rand() < self.epsilon:
            action_idx = np.random.choice(list(action_space.keys()))
        else:
            with torch.no_grad():
                q_values = self.model(state)  # shape: (1, output_dim)
                q_values = q_values[:, :len(neighbors)]  # consider only current valid actions
                action_idx = torch.argmax(q_values).item()

        return action_space[action_idx]  # return actual node index

    def load(self, path):
        """
        Loads model weights from file
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def save(self, path):
        """
        Saves model weights to file
        """
        torch.save(self.model.state_dict(), path)

    def eval_mode(self):
        self.model.eval()

    def train_mode(self):
        self.model.train()
