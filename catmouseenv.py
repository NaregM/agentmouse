import numpy as np

import numpy.typing as npt
from typing import List, Dict, Tuple, Optional

from gamegraph import GameGraph

class CatMouseEnv:
    """
    
    """
    def __init__(self, mode: str = "static", max_steps: int = 15) -> None:
        
        self.graph = GameGraph()
        self.nodes = list(self.graph.keys())
        self.max_steps = max_steps
        self.reset()
        
    def reset(self) -> npt.NDArray:
        """
        """
        return None