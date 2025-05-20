"""Cost-based search algorithms implementation."""

import heapq
from typing import Callable, List

from .graph import Graph


def ucs(
    graph: Graph, start: str, goal: str, cost_fn: Callable[[str, str], float]
) -> List[str]:
    """Perform Uniform Cost Search to find the lowest-cost path from start to goal.

    Time complexity: O((V + E) * log(V)) where V is vertices and E is edges
    Space complexity: O(V) for the priority queue, visited set, and path reconstruction

    Args:
        graph: The graph to search in
        start: Starting node ID
        goal: Goal node ID
        cost_fn: Function that returns the cost between two adjacent nodes

    Returns:
        A list of node IDs representing the lowest-cost path from start to goal
    """
    # Priority queue entries are (priority, cost_so_far, node, path)
    frontier = [(0, 0, start, [start])]
    best_cost = {start: 0}  # Track best known cost to each node

    while frontier:
        _, cost_so_far, node, path = heapq.heappop(frontier)

        # Skip if we've found a better path to this node
        if cost_so_far > best_cost[node]:
            continue

        # Found goal
        if node == goal:
            return path

        # Explore neighbors
        for neighbor in graph.neighbors(node):
            new_cost = cost_so_far + cost_fn(node, neighbor)

            # Only add to frontier if this is the best path so far
            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                new_path = [*path, neighbor]
                heapq.heappush(frontier, (new_cost, new_cost, neighbor, new_path))

    return []  # No path found


def a_star(
    graph: Graph,
    start: str,
    goal: str,
    cost_fn: Callable[[str, str], float],
    heuristic_fn: Callable[[str, str], float]
) -> List[str]:
    """Perform A* Search to find the optimal path from start to goal.

    Time complexity: O((V + E) * log(V)) where V is vertices and E is edges
    Space complexity: O(V) for the priority queue, visited set, and path reconstruction

    Args:
        graph: The graph to search in
        start: Starting node ID
        goal: Goal node ID
        cost_fn: Function that returns the cost between two adjacent nodes
        heuristic_fn: Admissible heuristic function estimating cost to goal

    Returns:
        A list of node IDs representing the optimal path from start to goal
    """
    # Priority queue entries are (priority, cost_so_far, node, path)
    frontier = [(heuristic_fn(start, goal), 0, start, [start])]
    best_cost = {start: 0}  # Track best known cost to each node

    while frontier:
        _, cost_so_far, node, path = heapq.heappop(frontier)

        # Skip if we've found a better path to this node
        if cost_so_far > best_cost[node]:
            continue

        # Found goal
        if node == goal:
            return path

        # Explore neighbors
        for neighbor in graph.neighbors(node):
            # Assuming unit cost between nodes if no cost_fn provided
            new_cost = cost_so_far + cost_fn(node, neighbor)

            # Only add to frontier if this is the best path so far
            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                new_path = [*path, neighbor]
                priority = new_cost + heuristic_fn(neighbor, goal)
                heapq.heappush(frontier, (priority, new_cost, neighbor, new_path))

    return []  # No path found
