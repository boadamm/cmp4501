import pytest

from sudoku.board import SudokuBoard

MIN_BOARD_STR_LEN = 10


def test_board_copy_and_solved():
    grid = [
        [1, 2, 3, 4, 5, 6],
        [None] * 6,
        [None] * 6,
        [None] * 6,
        [None] * 6,
        [None] * 6,
    ]
    board = SudokuBoard(grid)
    assert not board.is_solved()
    clone = board.copy()
    assert clone.grid == board.grid
    # fill remaining with 1-6 just for coverage
    for r in range(1, 6):
        for c in range(6):
            clone.grid[r][c] = ((r + c) % 6) + 1
    assert clone.is_solved()


def test_board_str():
    board = SudokuBoard([[None] * 6 for _ in range(6)])
    s = str(board)
    assert isinstance(s, str) and len(s) > MIN_BOARD_STR_LEN


def test_board_invalid_dimensions():
    """Test SudokuBoard initialization with invalid dimensions."""
    with pytest.raises(ValueError, match="Grid must have 6 rows"):
        SudokuBoard([[None] * 6] * 5)  # 5 rows instead of 6

    with pytest.raises(ValueError, match="Each row must have 6 cells"):
        grid_invalid_row = [[None] * 6 for _ in range(5)]
        grid_invalid_row.append([None] * 5)  # Last row has 5 cells
        SudokuBoard(grid_invalid_row)


def test_board_invalid_cell_values():
    """Test SudokuBoard initialization with invalid cell values."""
    grid = [[None] * 6 for _ in range(6)]
    grid[0][0] = 7  # Invalid symbol (7 not in 1-6)
    with pytest.raises(
        ValueError, match=r"Cells must be None or integers in \(1, 2, 3, 4, 5, 6\)"
    ):
        SudokuBoard(grid)

    grid[0][0] = 0  # Invalid symbol (0 not in 1-6)
    with pytest.raises(
        ValueError, match=r"Cells must be None or integers in \(1, 2, 3, 4, 5, 6\)"
    ):
        SudokuBoard(grid)

    grid[0][0] = "a"  # Invalid type
    with pytest.raises(
        ValueError, match=r"Cells must be None or integers in \(1, 2, 3, 4, 5, 6\)"
    ):
        SudokuBoard(grid)


def test_board_representation():
    # For 6x6 board, string representation should be non-trivial
    board = SudokuBoard([[None] * 6 for _ in range(6)])
    s = str(board)
    assert isinstance(s, str) and len(s) > MIN_BOARD_STR_LEN
