import numpy as np

import numpy.typing as npt
from typing import List, Dict, Tuple, Optional

from gamegraph import GameGraph
from models import Cat, Mouse

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

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
        """
        """
        cat_vec = np.zeros(len(self.nodes))
        mouse_vec = np.zeros(len(self.nodes))
        
        # Update the value where cat/mouse is currently located
        cat_vec[self.cat.position] = 1
        mouse_vec[self.mouse.position] = 1
        
        vec_combined = np.concatenate([mouse_vec, cat_vec])
        
    def step(self, mouse_action: int) -> Tuple[npt.NDArray, float, bool]:
        """
        """
        if mouse_action not in self.graph[self.mouse.position]:
            raise ValueError("Invalid action: not a neighbor of mouse!")
        
        # Mouse moves first
        self.mouse.position = mouse_action
        
        # Cat moves next (random policy for now)
        cat_moves = self.graph[self.cat.position]  # Available moves to cat
        self.cat.position = np.random.choice(cat_moves)
        
        self.steps += 1 # Game has a max allowed moves
        
        if self.cat.position == self.mouse.position:
            
            return self._get_state(), -1, True # Mouse loses
        
        elif self.steps >= self.max_steps:
            
            return self._get_state(), 1, True  # Mouse survives after max_steps moves
        
        else:
            
            return self._get_state(), -0.01, False # small step penalty, game continues
        
    def available_mouse_actions(self) -> List[int]:
        """
        """
        return self.graph[self.mouse.position]
    
    def render(self):
        """
        """
        print(f"Mouse position: {self.mouse.position}, Cat position: {self.cat.position}")
        
        