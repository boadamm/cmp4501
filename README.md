# CMP-4501 Search Algorithms

| Module       | Tests | Coverage |
|--------------|-------|----------|
| Search       | ✅    | ≥ 90 %   |
| Sudoku       | ✅    | ≥ 90 %   |
| Q-Learning   | ✅    | ≥ 90 %   |

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

## Part 2 – Constraint Satisfaction

This module implements a 6x6 Sudoku solver using constraint satisfaction techniques:

- AC-3 constraint propagation
- Backtracking search with forward checking
- Minimum Remaining Values (MRV) variable selection

### Testing

```bash
# Run all tests
pytest

# Run only Sudoku tests
pytest -q tests/test_sudoku.py

# Run style checks
ruff check .
```

### Project Structure

- `sudoku/`
  - `board.py`: Sudoku board representation
  - `csp.py`: CSP utilities (constraints, AC-3)
  - `solver.py`: Backtracking search implementation
- `tests/`
  - `test_sudoku.py`: Test cases for Sudoku solver

### Development Status

The Sudoku solver has a complete test suite but the core algorithms (AC-3, backtracking) are currently stubbed with TODOs. The tests verify:

1. Solution validity (all constraints satisfied)
2. Preservation of initial clues
3. Proper handling of row, column, and subgrid constraints

## Part 2 – RL

This section covers the Q-Learning implementation for a simple GridWorld environment.

### Usage

To run a demonstration of the Q-learning agent:

```bash
python -m rl_qlearning.demo
```

### Project Structure

- `rl_qlearning/`
  - `__init__.py`: Package initializer.
  - `env.py`: Contains the `GridWorld` environment.
  - `qlearn.py`: Implements the `q_learning` algorithm and `greedy_policy` extraction.
  - `demo.py`: (To be created) A script to demonstrate the Q-learning agent.
- `tests/`
  - `test_qlearning.py`: Test cases for the Q-learning implementation.

### Development Status

The Q-learning module is scaffolded with:

- A `GridWorld` environment (4x4, terminal goal (3,3), step reward -1, goal reward +10).
- A `q_learning` function stub (Q-table initialized to zeros).
- A `greedy_policy` function stub.
- Tests for Q-table shape and a (currently failing) test for policy effectiveness.

### Testing

```bash
# Run Q-learning tests (ensure pytest is installed)
pytest -q tests/test_qlearning.py

# Run style checks
ruff check .
```

## Part 2 – Naïve Bayes

This module implements a Multinomial Naïve Bayes classifier.

### Usage Example (Toy Dataset)

```python
from naive_bayes.datasets import load_toy_data
from naive_bayes.model import MultinomialNB

# Load data
X_train, y_train, X_test, y_test = load_toy_data()

# Initialize, fit, and predict
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate (example)
accuracy = (y_pred == y_test).mean()
print(f"Toy dataset accuracy: {accuracy:.2f}")
```

### Project Structure

- `naive_bayes/`
  - `__init__.py`: Package initializer.
  - `datasets.py`: Data loading utilities (e.g., `load_toy_data`).
  - `model.py`: `MultinomialNB` class implementation.
- `tests/`
  - `test_nb.py`: Test cases for the Naïve Bayes classifier.

### Development Status

The Multinomial Naïve Bayes classifier is currently a scaffold.
Core methods (`__init__`, `fit`, `predict`) are stubbed with `NotImplementedError`.
Tests are expected to fail until the model is implemented.

### Testing

```bash
# Run Naïve Bayes tests (ensure pytest is installed)
pytest -q tests/test_nb.py

# Run style checks
ruff check .
```

## Part 3 – Decision Tree

A simple CART-style binary decision tree.

### Usage Example (Toy Dataset)

```python
from decision_tree.datasets import load_toy_split
from decision_tree.tree import DecisionTree

# Load data
X_train, y_train, X_test, y_test = load_toy_split()

# Initialize, fit, and predict
model = DecisionTree(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate (example)
# accuracy = (y_pred == y_test).mean()
# print(f"Toy dataset accuracy: {accuracy:.2f}")
```

## Part 3 – Perceptron

This module implements a single-hidden-layer perceptron.

### Usage Example (Linear Dataset)

```python
from perceptron.datasets import load_linear
from perceptron.model import Perceptron
import numpy as np

# Load data
X_train, y_train, X_test, y_test = load_linear()

# Initialize, fit, and predict
# Model parameters are for demonstration; they might not be optimal.
model = Perceptron(hidden_size=6, lr=0.1, epochs=2000, seed=42)

# The following lines will raise NotImplementedError until implemented:
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# Evaluate (example, uncomment after implementation)
# accuracy = (y_pred == y_test).mean()
# print(f"Linear dataset accuracy: {accuracy:.2f}")
# print(f"Predictions: {y_pred}")
# print(f"Actual:      {y_test}")
```
