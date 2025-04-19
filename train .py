import numpy as np
from typing import Dict
import torch

from collections import deque
from catmouseenv import CatMouseEnv
from dqn import build_model, build_optimizer

def train_agent(config: Dict):
    """
    Trains a DQN agent to play either the mouse or the cat (mouse should always go first?)
    """
    return