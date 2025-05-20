"""
Sudoku solver implementation using CSP with MRV and forward checking.
"""

from copy import deepcopy
from typing import List, Optional, Union

from .board import SudokuBoard
from .csp import ac3, all_units, peers_map

Grid = List[List[Optional[int]]]


def solve(puzzle: Union[Grid, SudokuBoard]) -> Optional[Grid]:
    """Solve a Sudoku puzzle using a backtracking algorithm with MRV and AC-3.

    The puzzle can be provided as a SudokuBoard object or a raw grid (list of lists).
    The solver uses the Minimum Remaining Values (MRV) heuristic to select
    the next variable to assign and AC-3 for constraint propagation (forward checking).

    Args:
        puzzle (Union[Grid, SudokuBoard]): The Sudoku puzzle to solve.
            It can be a SudokuBoard instance or a 2D list representing the grid,
            where None or 0 represents an empty cell.

    Returns:
        Optional[Grid]: A 2D list representing the solved Sudoku grid if a solution
        is found, otherwise None.
    """
    # normalise
    board = puzzle if isinstance(puzzle, SudokuBoard) else SudokuBoard(puzzle)
    size = board.size
    units = all_units(size)
    peers = peers_map(units)

    # initial domains
    domains = {
        (r, c): (
            {board.grid[r][c]}
            if board.grid[r][c] is not None
            else set(range(1, size + 1))
        )
        for r in range(size)
        for c in range(size)
    }

    # propagate clues
    if not ac3(domains, peers):
        return None

    def is_solved(dom):
        return all(len(v) == 1 for v in dom.values())

    def select_mrv(dom):
        return min((c for c in dom if len(dom[c]) > 1), key=lambda c: len(dom[c]))

    def backtrack(dom) -> Optional[dict]:
        if is_solved(dom):
            return dom
        cell = select_mrv(dom)
        for v in sorted(dom[cell]):
            new_dom = deepcopy(dom)
            new_dom[cell] = {v}
            # forward-check: remove v from peers
            valid = True
            for p in peers[cell]:
                if v in new_dom[p]:
                    new_dom[p].remove(v)
                    if not new_dom[p]:
                        valid = False
                        break
            if valid and ac3(new_dom, peers):
                result = backtrack(new_dom)
                if result:
                    return result
        return None

    result_dom = backtrack(domains)
    if result_dom is None:
        return None
    return [[next(iter(result_dom[(r, c)])) for c in range(size)] for r in range(size)]
