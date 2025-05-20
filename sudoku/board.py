"""
Sudoku board representation and utility methods.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SudokuBoard:
    """A 6x6 Sudoku board representation."""
    grid: List[List[Optional[int]]]
    size: int = 6
    symbols: range = range(1, 7)

    def __post_init__(self) -> None:
        """Validate board dimensions."""
        if len(self.grid) != self.size:
            raise ValueError(f"Grid must have {self.size} rows")
        if any(len(row) != self.size for row in self.grid):
            raise ValueError(f"Each row must have {self.size} cells")
        if any(
            not (cell is None or (isinstance(cell, int) and cell in self.symbols))
            for row in self.grid
            for cell in row
        ):
            raise ValueError(f"Cells must be None or integers in {self.symbols}")

    def is_solved(self) -> bool:
        """Return True if the board is completely filled and valid."""
        return all(
            isinstance(cell, int) and cell in self.symbols
            for row in self.grid
            for cell in row
        )

    def copy(self) -> 'SudokuBoard':
        """Return a deep copy of the board."""
        return SudokuBoard([row[:] for row in self.grid])

    def __str__(self) -> str:
        """Return a string representation of the board."""
        lines = []
        for i, row in enumerate(self.grid):
            if i > 0 and i % 2 == 0:
                lines.append('-' * 13)
            row_str = '|'.join(
                ' '.join(str(cell or '.') for cell in row[j:j+3])
                for j in (0, 3)
            )
            lines.append(row_str)
        return '\n'.join(lines) 