"""
Sudoku solver implementation using CSP with MRV and forward checking.
"""
from copy import deepcopy
from typing import Union, Optional, List
from .csp import all_units, peers_map, ac3
from .board import SudokuBoard

Grid = List[List[Optional[int]]]

def solve(puzzle: Union[Grid, SudokuBoard]) -> Optional[Grid]:
    # normalise
    board = puzzle if isinstance(puzzle, SudokuBoard) else SudokuBoard(puzzle)
    size = board.size
    units = all_units(size)
    peers = peers_map(units)

    # initial domains
    domains = {(r, c): ({board.grid[r][c]} if board.grid[r][c] is not None else set(range(1, size+1)))
               for r in range(size) for c in range(size)}

    # propagate clues
    if not ac3(domains, peers):
        return None

    def is_solved(dom): return all(len(v) == 1 for v in dom.values())

    def select_mrv(dom):
        return min((c for c in dom if len(dom[c]) > 1), key=lambda c: len(dom[c]))

    def backtrack(dom) -> Optional[dict]:
        if is_solved(dom):
            return dom
        cell = select_mrv(dom)
        for v in sorted(dom[cell]):
            new_dom = deepcopy(dom)
            new_dom[cell] = {v}
            # forward‑check: remove v from peers
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