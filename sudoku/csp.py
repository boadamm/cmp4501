"""
Constraint Satisfaction Problem (CSP) utilities for Sudoku.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sudoku.grid import Cell

# Type definition for a cell coordinate
Cell = tuple[int, int]  # (row, col) helper alias

BOX_H, BOX_W = 2, 3  # 6x6 → 6 2x3 boxes


def all_units(size: int = 6) -> list[list[Cell]]:
    """Return 6 row units, 6 column units, and 6 (2x3) box units.

    Args:
        size (int, optional): The size of the Sudoku grid. Defaults to 6.

    Returns:
        list[list[Cell]]: A list of units, where each unit is a list of cells.
    """
    # Row units - each row is a unit
    rows = [[(r, c) for c in range(size)] for r in range(size)]

    # Column units - each column is a unit
    cols = [[(r, c) for r in range(size)] for c in range(size)]

    # Box units - each 2x3 box is a unit
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

    Args:
        units (list[list[Cell]]): A list of all units (rows, columns, boxes).

    Returns:
        dict[Cell, set[Cell]]: A dictionary mapping each cell to a set of its
                               peer cells.
    """
    peers: dict[Cell, set[Cell]] = {}

    # First, identify which units each cell belongs to
    cell_to_units_map: dict[Cell, list[list[Cell]]] = {}
    for unit in units:
        for cell in unit:
            if cell not in cell_to_units_map:
                cell_to_units_map[cell] = []
            cell_to_units_map[cell].append(unit)

    # Then build peers using the cell_to_units_map
    for cell, units_for_this_cell in cell_to_units_map.items():
        peers[cell] = set()
        for unit in units_for_this_cell:
            for peer_cell in unit:
                if peer_cell != cell:
                    peers[cell].add(peer_cell)
    return peers


def revise(domains: dict[Cell, set[int]], xi: Cell, xj: Cell) -> bool:
    """
    For an all-different (≠) constraint, remove v from Xi if
    Xj is a *singleton* {v}.  Return True iff we removed something.

    Args:
        domains (dict[Cell, set[int]]): A dictionary mapping cells to their possible
                                        values.
        xi (Cell): The first cell in the constraint.
        xj (Cell): The second cell in the constraint.

    Returns:
        bool: True if a value was removed from the domain of xi, False otherwise.
    """
    if len(domains[xj]) != 1:
        return False
    (vj,) = (next(iter(domains[xj])),)
    if vj in domains[xi]:
        # This is the version from the initial prompt
        domains[xi].remove(vj)
        return True
    return False


def ac3(domains: dict[Cell, set[int]], peers: dict[Cell, set[Cell]]) -> bool:
    """
    Establish arc consistency for the given domains and peer relationships.

    This function implements the AC-3 algorithm to reduce the domains of variables
    (cells) by ensuring that for every pair of constrained variables (xi, xj),
    every value in the domain of xi has a corresponding consistent value in the
    domain of xj.

    Args:
        domains (dict[Cell, set[int]]): A dictionary mapping each cell to its current
            set of possible values (its domain).
        peers (dict[Cell, set[Cell]]): A dictionary mapping each cell to the set of
            its peer cells (cells that share a unit with it).

    Returns:
        bool: True if arc consistency is achieved and no domain is empty,
              False if an inconsistency is found (a domain becomes empty).
    """
    from collections import deque

    queue = deque((xi, xj) for xi in domains for xj in peers[xi])
    while queue:
        xi, xj = queue.popleft()
        if revise(domains, xi, xj):
            if not domains[xi]:
                return False  # domain wiped ⇒ inconsistency
            for xk in peers[xi]:
                if xk != xj:  # avoid Xi→Xi
                    queue.append((xk, xi))
    return True
