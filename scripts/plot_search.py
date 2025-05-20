# pragma: no cover
"""
BFS vs A* — pop-count with explicit Manhattan heuristic.
Outputs: figures/search_nodes.png
"""

import heapq
from collections import deque

import matplotlib.pyplot as plt

from search.graph import grid_graph


def manhattan_lambda(goal_row: int, goal_col: int):
    """Return h(n) that parses 'r,c' → Manhattan to goal."""

    def h(node: str) -> int:
        r, c = map(int, node.split(","))
        return abs(goal_row - r) + abs(goal_col - c)

    return h


def bfs_expansions(graph, start: str, goal: str) -> int:
    frontier, explored, count = deque([start]), set(), 0
    parent = {start: None}
    while frontier:
        node = frontier.popleft()
        if node in explored:
            continue
        explored.add(node)
        count += 1
        if node == goal:
            break
        for nbr in graph.neighbors(node):
            if nbr not in parent:
                parent[nbr] = node
                frontier.append(nbr)
    return count


def astar_expansions(graph, start: str, goal: str) -> int:
    goal_r, goal_c = map(int, goal.split(","))
    h = manhattan_lambda(goal_r, goal_c)

    frontier = [(h(start), 0, start)]
    g_cost = {start: 0}
    explored, count = set(), 0

    while frontier:
        f, g, node = heapq.heappop(frontier)
        if node in explored:
            continue
        explored.add(node)
        count += 1
        if node == goal:
            break
        for nbr in graph.neighbors(node):
            new_g = g + 1
            if new_g < g_cost.get(nbr, 1e9):
                g_cost[nbr] = new_g
                heapq.heappush(frontier, (new_g + h(nbr), new_g, nbr))
    return count


sizes, bfs_nodes, astar_nodes = [], [], []
for n in range(5, 26, 5):
    g = grid_graph(n, n)
    s, t = "0,0", f"{n - 1},0"
    sizes.append(n)
    bfs_nodes.append(bfs_expansions(g, s, t))
    astar_nodes.append(astar_expansions(g, s, t))

plt.plot(sizes, bfs_nodes, label="BFS")
plt.plot(sizes, astar_nodes, label="A*")
plt.yscale("log")
plt.xlabel("Grid size n")
plt.ylabel("# nodes expanded (log)")
plt.title("BFS vs A* node expansions")
plt.legend()
plt.savefig("figures/search_nodes.png", dpi=200)
print("saved → figures/search_nodes.png")
