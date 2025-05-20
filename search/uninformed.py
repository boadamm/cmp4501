"""Uninformed search algorithms implementation."""
from collections import deque
from typing import List

from .graph import Graph


def dfs(graph: Graph, start: str, goal: str) -> List[str]:
    """Perform Depth-First Search to find a path from start to goal.
    
    Uses an explicit stack (LIFO list) to avoid recursion depth issues.
    
    Time complexity: O(V + E) where V is number of vertices and E is number of edges
    Space complexity: O(V) for the visited set and path reconstruction
    
    Args:
        graph: The graph to search in
        start: Starting node ID
        goal: Goal node ID
    
    Returns:
        A list of node IDs representing the path from start to goal.
        Returns an empty list if no path exists.
    """
    if start == goal:
        return [start]
        
    # Stack contains tuples of (node, path_to_node)
    stack: List[tuple[str, List[str]]] = [(start, [start])]
    visited = {start}
    
    while stack:
        node, path = stack.pop()  # LIFO for DFS
        
        for neighbor in graph.neighbors(node):
            if neighbor == goal:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    
    return []  # No path found


def bfs(graph: Graph, start: str, goal: str) -> List[str]:
    """Perform Breadth-First Search to find a path from start to goal.
    
    Uses a deque as an efficient queue data structure.
    Guarantees the shortest path in terms of number of edges.
    
    Time complexity: O(V + E) where V is number of vertices and E is number of edges
    Space complexity: O(V) for the queue, visited set, and path reconstruction
    
    Args:
        graph: The graph to search in
        start: Starting node ID
        goal: Goal node ID
    
    Returns:
        A list of node IDs representing the shortest path from start to goal.
        Returns an empty list if no path exists.
    """
    if start == goal:
        return [start]
        
    # Queue contains tuples of (node, path_to_node)
    queue: deque[tuple[str, List[str]]] = deque([(start, [start])])
    visited = {start}
    
    while queue:
        node, path = queue.popleft()  # FIFO for BFS
        
        for neighbor in graph.neighbors(node):
            if neighbor == goal:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path found 