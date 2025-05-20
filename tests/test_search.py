"""Test cases for search algorithms."""

from typing import Callable, List

import pytest

from search.cost_search import a_star, ucs
from search.graph import Graph, grid_graph
from search.heuristics import manhattan
from search.uninformed import bfs, dfs

MAX_EXPLORED_IN_4X4_GRID = 16


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


@pytest.mark.parametrize(
    "algorithm",
    [
        dfs,
        bfs,
        lambda g, s, e: ucs(g, s, e, lambda n1, n2: 1.0),
        lambda g, s, e: a_star(g, s, e, lambda n1, n2: 1.0, manhattan),
    ],
)
def test_search_finds_valid_path(grid_4x4: Graph, algorithm):
    """Test that each algorithm finds a valid path from start to goal."""
    start, goal = "0,0", "3,3"
    path = algorithm(grid_4x4, start, goal)

    assert path[0] == start
    assert path[-1] == goal

    # Verify path continuity
    for i in range(len(path) - 1):
        assert path[i + 1] in grid_4x4.neighbors(path[i])


@pytest.mark.parametrize(
    "optimal_algorithm",
    [
        bfs,  # Optimal for unweighted graphs
        lambda g, s, e: ucs(g, s, e, lambda n1, n2: 1.0),
        lambda g, s, e: a_star(g, s, e, lambda n1, n2: 1.0, manhattan),
    ],
)
def test_optimal_path_length(grid_4x4: Graph, optimal_algorithm):
    """Test that BFS, UCS, and A* find paths of minimal length."""
    start, goal = "0,0", "3,3"
    optimal_path = bfs(grid_4x4, start, goal)  # Use BFS as reference
    test_path = optimal_algorithm(grid_4x4, start, goal)

    assert len(test_path) == len(optimal_path)


@pytest.mark.parametrize(
    "algorithm",
    [
        dfs,
        bfs,
        lambda g, s, e: ucs(g, s, e, lambda n1, n2: 1.0),
        lambda g, s, e: a_star(g, s, e, lambda n1, n2: 1.0, manhattan),
    ],
)
def test_explored_nodes_limit(grid_4x4: Graph, algorithm, monkeypatch):
    """Test that algorithms don't explore more nodes than exist in the graph."""
    start, goal = "0,0", "3,3"
    explored = count_explored_nodes(monkeypatch, grid_4x4, algorithm, start, goal)

    assert explored <= MAX_EXPLORED_IN_4X4_GRID  # 4x4 grid has 16 nodes


@pytest.mark.parametrize(
    "algorithm",
    [
        dfs,
        bfs,
        lambda g, s, e: ucs(g, s, e, lambda n1, n2: 1.0),
        lambda g, s, e: a_star(g, s, e, lambda n1, n2: 1.0, manhattan),
    ],
)
def test_start_equals_goal(grid_4x4: Graph, algorithm):
    """Test that each algorithm handles the case where start equals goal."""
    node = "0,0"
    path = algorithm(grid_4x4, node, node)
    assert path == [node]


@pytest.mark.parametrize("algorithm", [dfs, bfs])
def test_no_path_found(algorithm):
    """Test that DFS and BFS return an empty list when no path exists."""
    graph = Graph()
    graph.add_edge("A", "B")
    graph.add_edge("C", "D")
    path = algorithm(graph, "A", "D")
    assert path == []


def test_cost_search_no_path_found():
    """Test that UCS and A* return an empty list when no path exists."""
    graph = Graph()
    graph.add_edge("A", "B")
    graph.add_edge("C", "D")

    def cost_fn_ucs_no_path(n1, n2):
        return 1.0

    def heuristic_fn_ucs_no_path(n1, n2):
        return 0.0

    assert ucs(graph, "A", "D", cost_fn_ucs_no_path) == []
    assert a_star(graph, "A", "D", cost_fn_ucs_no_path, heuristic_fn_ucs_no_path) == []


def test_cost_search_skip_suboptimal_path():
    """Test that UCS and A* skip paths that are found later with higher cost."""
    graph = Graph()
    graph.add_edge("S", "A")
    graph.add_edge("S", "B")
    graph.add_edge("A", "G")
    graph.add_edge("B", "G")
    graph.add_edge("S", "G") # Direct but higher cost initially in frontier

    # UCS Test
    # S->A (1), S->B (1), A->G (1), B->G(100)
    # S->G (10) but added to frontier later than S->A and S->B might explore it
    # We want to ensure S->A->G (cost 2) is chosen over S->G (cost 10)
    # and also that if G is reached via S->B->G (cost 101) it's ignored
    # if S->A->G (2) found
    costs_ucs = {("S","A"):1, ("S","B"):1, ("A","G"):1, ("B","G"):100, ("S","G"):10}
    # Symmetric for graph.py's undirected edges
    for (u,v),c in list(costs_ucs.items()):
        costs_ucs[(v,u)] = c

    def cost_fn_ucs(n1, n2):
        return costs_ucs.get((n1,n2), float('inf'))

    path_ucs = ucs(graph, "S", "G", cost_fn_ucs)
    assert path_ucs == ["S", "A", "G"]

    # A* Test: Heuristic makes S->G look good, but path S->A->G is better
    # S->A (1), S->B (1), A->G (1), B->G (1)
    # S->G (1) - if cost_so_far + h is used, direct S->G might be preferred.
    # Heuristic h(X,G): S=1, A=0, B=10, G=0
    # Frontier:
    # (h(S,G)+0, 0, S, [S]) -> (1,0,S,[S])
    # Pop S: Neighbors A, B, G
    # A: cost S->A = 1.  h(A,G)=0. Total S->A = 1. Push (1+0, 1, A, [S,A])
    # B: cost S->B = 1.  h(B,G)=10. Total S->B = 1. Push (1+10, 1, B, [S,B])
    # G: cost S->G = 10. h(G,G)=0. Total S->G = 10. Push (10+0, 10, G, [S,G])
    # Pop A: Neighbors S, G
    # G: cost S->A->G = 1+1=2. h(G,G)=0. Total = 2. Push (2+0, 2, G, [S,A,G])
    # Path [S,A,G] should be found.
    # The continue should be hit for the S->G path if explored later.
    costs_astar = {("S","A"):1, ("S","B"):1, ("A","G"):1, ("B","G"):1, ("S","G"):10}
    for (u,v),c in list(costs_astar.items()):
        costs_astar[(v,u)] = c

    def cost_fn_astar(n1, n2):
        return costs_astar.get((n1,n2), float('inf'))

    def heuristic_fn_astar(n,g):
        # Heuristic values for A* an example
        heuristic_values = {"S":1, "A":0, "B":10, "G":0}
        return heuristic_values.get(n, float('inf'))

    path_astar = a_star(graph, "S", "G", cost_fn_astar, heuristic_fn_astar)
    assert path_astar == ["S", "A", "G"]


def test_search_complex_graph_ucs_astar():
    """Test UCS and A* on a more complex graph with specific costs/heuristics."""
    graph = Graph()
    nodes = ["S", "A", "B", "C", "D", "G"]
    for node in nodes:
        graph.add_node(node)
    # Edges for UCS and A*
    # UCS: S->A(1), S->B(1), A->G(1), B->G(100), S->G(10)
    # Optimal UCS: S->A->G (cost 2)
    # A*: S->A(1), S->B(1), A->G(1), B->G(1), S->G(10)
    # Heuristic h(S,G)=1, h(A,G)=0, h(B,G)=10, h(G,G)=0 (underestimate for B->G)
    # Optimal A*: S->A->G (cost 2, total f=2+0=2)
    # Path S->B->G (cost 2, total f=2+0=2) might be found first if
    # tie-broken by path length
    # Path S->G (cost 10, total f=10+0=10)

    # S->G (10) but added to frontier later than S->A and S->B might explore it
    # We want to ensure S->A->G (cost 2) is chosen over S->G (cost 10)
    # and also that if G is reached via S->B->G (cost 101) it's ignored
    # if S->A->G (2) found
    costs_ucs = {("S","A"):1, ("S","B"):1, ("A","G"):1, ("B","G"):100, ("S","G"):10}
    # Symmetric for graph.py's undirected edges
    for (u,v),c in list(costs_ucs.items()):
        costs_ucs[(v,u)] = c

    def cost_fn_ucs(n1, n2):
        return costs_ucs.get((n1,n2), float('inf'))

    path_ucs = ucs(graph, "S", "G", cost_fn_ucs)
    assert path_ucs == ["S", "A", "G"]

    # A* Test
    # Frontier: (f, cost, node, path)
    # Start S: Push (1+1, 0, S, [S])
    # Pop S: Neighbors A, B, G
    # A: cost S->A = 1. h(A,G)=0. Total = 1. Push (1+0, 1, A, [S,A])
    # B: cost S->B = 1. h(B,G)=10. Total = 11. Push (1+10, 1, B, [S,B])
    # G: cost S->G = 10. h(G,G)=0. Total = 10. Push (10+0, 10, G, [S,G])
    # Pop A: Neighbors S, G
    # G: cost S->A->G = 1+1=2. h(G,G)=0. Total = 2. Push (2+0, 2, G, [S,A,G])
    # Path [S,A,G] should be found.
    # The continue should be hit for the S->G path if explored later.
    costs_astar = {("S","A"):1, ("S","B"):1, ("A","G"):1, ("B","G"):1, ("S","G"):10}
    for (u,v),c in list(costs_astar.items()):
        costs_astar[(v,u)] = c

    def cost_fn_astar(n1, n2):
        return costs_astar.get((n1,n2), float('inf'))

    def heuristic_fn_astar(n,g):
        return {"S":1, "A":0, "B":10, "G":0}.get(n, float('inf'))

    path_astar = a_star(graph, "S", "G", cost_fn_astar, heuristic_fn_astar)
    assert path_astar == ["S", "A", "G"]
