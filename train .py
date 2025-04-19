import numpy as np
from typing import Dict
import torch

from collections import deque
from tqdm import tqdm

from catmouseenv import CatMouseEnv
from dqn import build_model, build_optimizer

def train_agent(config: Dict):
    """
    Trains a DQN agent to play either the mouse or the cat (mouse should always go first?)
    """
    input_dim = config.get("input_dim")
    hidden_dims = config.get("hidden_dims")
    output_dim = config.get("output_dim")
    learning_rate = config.get("learning_rate", 1e-3)
    epochs = config.get("epochs", 1_000)
    batch_size = config.get("batch_size", 64)
    memory_size = config.get("memory_size", 2_000)
    gamma = config.get("gamma", 0.9)
    epsilon_start = config.get("epsilon_start", 0.3)
    epsilon_end = config.get("epsilon_end", 0.05)
    epsilon_decay = config.get("epsilon_decay", 0.995)
    sync_freq = config.get("sync_freq", 100)
    model_path = config.get("model_path", "model.pt")
    role = config.get("role", "mouse")
    
    model, target_network = build_model(input_dim, hidden_dims, output_dim)
    optimizer = build_optimizer(model, learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    replay = deque(maxlen=memory_size)
    losses = []
    j = 0
    
    for epoch in tqdm(range(epochs)):
        
        pass
    
    return