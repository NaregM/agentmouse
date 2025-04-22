import torch
import torch.nn as nn

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),  # max number of neighbors
        )

    def forward(self, x):
        return self.net(x)
    
    


def build_model(input_dim, hidden_dims, output_dim):
    """
    Returns:
        Tuple[model, target_model]: two instances of the same DQN network with identical weights.
        The second model is intended to be used as a target network.
    """
    model = DQN(input_dim, hidden_dims, output_dim)
    target = DQN(input_dim, hidden_dims, output_dim)
    # Copy weights from model to target model (for stable Q-learning)
    target.load_state_dict(model.state_dict())

    return model, target


def build_optimizer(model: nn.Module, learning_rate: float = 1e-4):
    """ """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)
