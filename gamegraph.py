from typing import List, Dict


class GameGraph:
    """
    Represents the fixed game graph for the Cat and Mouse environment.
    Provides neighbor lookup for each node.
    """

    def __init__(self) -> None:

        # adjacency list
        self.adj: Dict[int, List[int]] = {
            0: [1, 4, 5],
            1: [0, 2, 3],
            2: [1, 4],
            3: [1, 5],
            4: [0, 2, 5],
            5: [0, 3, 4],
        }

    def neighbors(self, node: int) -> List[int]:
        """
        Return list of neighboring nodes for a given node.
        """
        return self.adj[node]
