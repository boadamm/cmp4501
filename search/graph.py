"""Graph implementation and utilities for search algorithms."""
from collections import defaultdict
from typing import Dict, List, Set


class Graph:
    """A graph represented using an adjacency list."""
    
    def __init__(self):
        """Initialize an empty graph."""
        self.edges: Dict[str, Set[str]] = defaultdict(set)
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an undirected edge between from_node and to_node."""
        self.edges[from_node].add(to_node)
        self.edges[to_node].add(from_node)
    
    def neighbors(self, node: str) -> List[str]:
        """Return a list of nodes connected to the given node."""
        return sorted(list(self.edges[node]))  # Sorted for deterministic behavior


def grid_graph(n_rows: int, n_cols: int) -> Graph:
    """Create a grid graph with the specified dimensions.
    
    Nodes are labeled as "row,col" strings (e.g., "0,0", "0,1", etc.).
    Each node is connected to its adjacent nodes (up, down, left, right).
    
    Args:
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
    
    Returns:
        A Graph instance representing the grid
    """
    graph = Graph()
    
    for row in range(n_rows):
        for col in range(n_cols):
            current = f"{row},{col}"
            
            # Add edges to adjacent nodes
            if row > 0:  # Up
                graph.add_edge(current, f"{row-1},{col}")
            if row < n_rows - 1:  # Down
                graph.add_edge(current, f"{row+1},{col}")
            if col > 0:  # Left
                graph.add_edge(current, f"{row},{col-1}")
            if col < n_cols - 1:  # Right
                graph.add_edge(current, f"{row},{col+1}")
    
    return graph 