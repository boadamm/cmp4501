"""Test cases for search algorithms."""
import pytest
from typing import Callable, List

from search.graph import Graph, grid_graph
from search.uninformed import dfs, bfs
from search.cost_search import ucs, a_star
from search.heuristics import manhattan


@pytest.fixture
def grid_4x4() -> Graph:
    """Create a 4x4 grid graph for testing."""
    return grid_graph(4, 4)


def count_explored_nodes(monkeypatch, graph: Graph, algorithm: Callable, *args) -> int:
    """Count number of nodes explored during search via neighbors() calls."""
    explored_count = 0
    original_neighbors = graph.neighbors

    def counting_neighbors(node: str) -> List[str]:
        nonlocal explored_count
        explored_count += 1
        return original_neighbors(node)

    monkeypatch.setattr(graph, "neighbors", counting_neighbors)
    algorithm(graph, *args)
    return explored_count


@pytest.mark.parametrize("algorithm", [
    dfs,
    bfs,
    lambda g, s, e: ucs(g, s, e, lambda n1, n2: 1.0),
    lambda g, s, e: a_star(g, s, e, manhattan)
])
def test_search_finds_valid_path(grid_4x4: Graph, algorithm):
    """Test that each algorithm finds a valid path from start to goal."""
    start, goal = "0,0", "3,3"
    path = algorithm(grid_4x4, start, goal)
    
    assert path[0] == start
    assert path[-1] == goal
    
    # Verify path continuity
    for i in range(len(path) - 1):
        assert path[i + 1] in grid_4x4.neighbors(path[i])


@pytest.mark.parametrize("optimal_algorithm", [
    bfs,  # Optimal for unweighted graphs
    lambda g, s, e: ucs(g, s, e, lambda n1, n2: 1.0),
    lambda g, s, e: a_star(g, s, e, manhattan)
])
def test_optimal_path_length(grid_4x4: Graph, optimal_algorithm):
    """Test that BFS, UCS, and A* find paths of minimal length."""
    start, goal = "0,0", "3,3"
    optimal_path = bfs(grid_4x4, start, goal)  # Use BFS as reference
    test_path = optimal_algorithm(grid_4x4, start, goal)
    
    assert len(test_path) == len(optimal_path)


@pytest.mark.parametrize("algorithm", [
    dfs,
    bfs,
    lambda g, s, e: ucs(g, s, e, lambda n1, n2: 1.0),
    lambda g, s, e: a_star(g, s, e, manhattan)
])
def test_explored_nodes_limit(grid_4x4: Graph, algorithm, monkeypatch):
    """Test that algorithms don't explore more nodes than exist in the graph."""
    start, goal = "0,0", "3,3"
    explored = count_explored_nodes(monkeypatch, grid_4x4, algorithm, start, goal)
    
    assert explored <= 16  # 4x4 grid has 16 nodes 


@pytest.mark.parametrize("algorithm", [
    dfs,
    bfs,
    lambda g, s, e: ucs(g, s, e, lambda n1, n2: 1.0),
    lambda g, s, e: a_star(g, s, e, manhattan)
])
def test_start_equals_goal(grid_4x4: Graph, algorithm):
    """Test that each algorithm handles the case where start equals goal."""
    node = "0,0"
    path = algorithm(grid_4x4, node, node)
    assert path == [node] 