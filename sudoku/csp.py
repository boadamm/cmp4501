"""
Constraint Satisfaction Problem (CSP) utilities for Sudoku.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sudoku.grid import Cell

# Type definition for a cell coordinate
Cell = tuple[int, int]     # (row, col) helper alias

BOX_H, BOX_W = 2, 3          # 6×6 → 6 2×3 boxes

def all_units(size: int = 6) -> list[list[Cell]]:
    """Return 6 row units, 6 column units, and 6 (2×3) box units."""
    # Row units - each row is a unit
    rows = [[(r, c) for c in range(size)] for r in range(size)]
    
    # Column units - each column is a unit
    cols = [[(r, c) for r in range(size)] for c in range(size)]
    
    # Box units - each 2×3 box is a unit
    boxes = []
    for br in range(0, size, BOX_H):  # Step by BOX_H (2)
        for bc in range(0, size, BOX_W):  # Step by BOX_W (3)
            box = []
            for r in range(br, br + BOX_H):
                for c in range(bc, bc + BOX_W):
                    box.append((r, c))
            boxes.append(box)
    
    units = rows + cols + boxes
    return units


def peers_map(units: list[list[Cell]]) -> dict[Cell, set[Cell]]:
    """Return dictionary mapping each cell to its peer cells (sharing a unit).
    
    A cell's peers are all other cells in the same row, column, or box.
    Each cell should have exactly 12 peers in a 6x6 Sudoku:
    - 5 from its row
    - 5 from its column
    - 2 from its box (not already counted from row/column)
    """
    peers: dict[Cell, set[Cell]] = {}
    
    # First, identify which units each cell belongs to
    cell_units: dict[Cell, list[list[Cell]]] = {}
    for unit in units:
        for cell in unit:
            if cell not in cell_units:
                cell_units[cell] = []
            cell_units[cell].append(unit)
    
    # Then build peers by first adding row and column peers
    for cell in cell_units:
        peers[cell] = set()
        # Find the units this cell belongs to
        for unit in cell_units[cell]:
            # Add all cells from this unit except the cell itself
            peers[cell].update(c for c in unit if c != cell)
    
    return peers


def revise(domains: dict[Cell, set[int]], xi: Cell, xj: Cell) -> bool:
    """
    For an all-different (≠) constraint, remove v from Xi if
    Xj is a *singleton* {v}.  Return True iff we removed something.
    """
    if len(domains[xj]) != 1:
        return False
    (vj,) = next(iter(domains[xj])),
    if vj in domains[xi]:
        # This is the version from the initial prompt
        domains[xi].remove(vj)
        return True
    return False


def ac3(domains: dict[Cell, set[int]], peers: dict[Cell, set[Cell]]) -> bool:
    from collections import deque
    queue = deque((xi, xj) for xi in domains for xj in peers[xi])
    while queue:
        xi, xj = queue.popleft()
        if revise(domains, xi, xj):
            if not domains[xi]:
                return False               # domain wiped ⇒ inconsistency
            for xk in peers[xi]:
                if xk != xj:               # avoid Xi→Xi
                    queue.append((xk, xi))
    return True