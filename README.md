# CMP-4501 Search Algorithms

## Part 1 – Search

This module implements various search algorithms for traversing graphs, including:

- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- Uniform Cost Search (UCS)
- A* Search

### Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install pytest pytest-cov ruff
```

### Testing

```bash
# Run tests with coverage
pytest

# Run style checks
ruff check .
```

### Project Structure

- `search/`
  - `graph.py`: Graph implementation and grid graph constructor
  - `uninformed.py`: DFS and BFS implementations
  - `cost_search.py`: UCS and A* implementations
  - `heuristics.py`: Heuristic functions (Manhattan distance)
- `tests/`
  - `test_search.py`: Test cases for all search algorithms

### Development Status

The search algorithms are currently stubbed out with TODO comments. The test suite is set up to verify:

1. Path validity (starts at start node, ends at goal node)
2. Path optimality for BFS, UCS, and A*
3. Search efficiency (explored nodes ≤ graph size)
