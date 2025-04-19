from typing import List

class GameGraph:
    """
    """
    def __init__(self) -> None:
        
        # adjacency list
        self.adj = {
            0: [1, 4, 5],
            1: [0, 2, 3],
            2: [1, 4],
            3: [1, 5],
            4: [0, 2, 5],
            5: [0, 3, 4]
        }
        
    def neighbors(self, node: int) -> List[int]:
        return self.adj[node]