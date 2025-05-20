"""Heuristic functions for informed search algorithms."""


def manhattan(node_a: str, node_b: str) -> float:
    """Calculate the Manhattan distance between two grid nodes.

    Nodes are expected to be in the format "row,col" (e.g., "3,4").

    Args:
        node_a: First node ID in "row,col" format
        node_b: Second node ID in "row,col" format

    Returns:
        The Manhattan distance between the nodes
    """
    row_a, col_a = map(int, node_a.split(","))
    row_b, col_b = map(int, node_b.split(","))

    return abs(row_b - row_a) + abs(col_b - col_a)
