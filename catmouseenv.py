import numpy as np

import numpy.typing as npt
from typing import List, Dict, Tuple, Optional

from gamegraph import GameGraph
from models import Cat, Mouse

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
        if self.mode == 'static':
            
            # For 'static' mode, cat and mouse start from a preset position
            self.cat   = Cat(position=0)
            self.mouse = Mouse(position=4)
            
        elif self.mode == "random":
            # For 'random' node cat and mouse are assigned initial loc on random 
            self.cat = Cat(position=np.random.choice(self.nodes))
            self.mouse = Mouse(position=np.random.choice([n for n in self.nodes if n!=self.cat.position]))
            
        else:
            
            raise ValueError("Mode must be either 'static' or 'random'!")
        
        self.steps = 0
        return self._get_state()
    
    def _get_state(self) -> npt.NDArray:
        
        return