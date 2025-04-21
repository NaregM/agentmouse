import numpy as np
import torch
import torch.nn as nn
from dqn import DQN
import numpy.typing as npt

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class DQNAgent:
    
    def __init__(self, input_dim, hidden_dims, output_dim, epsilon = 0.15) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_dim, hidden_dims, output_dim).to(self.device)
        self.epsilon = epsilon 
        
    def act(self, state, valid_actions, eval = False):
        """
        Selects an action using epsilon-greedy policy
        """
        if isinstance(state, npt.NDArray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.unsqueeze(0).to(self.device)
            
        if not eval and np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        
        with torch.no_grad():
            q_values = self.model(state)
            best_idx = torch.argmax(q_values[0, :len(valid_actions)]).item()
            return valid_actions[best_idx]
        
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
        