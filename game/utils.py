"""
This module provides utility functions for the EVE Hacking game.

It currently contains helper logic for translating UI events (like mouse clicks)
into game world coordinates.

Architecture Role:
    - Utilities: Shared helper functions that don't fit neatly into Model or View classes.

Modification History:
    - 2025-11-28: Created and moved get_node_at_mouse from main.py.
"""

from .models import System

def get_node_at_mouse(board: System, my: int, mx: int):
    """
    Converts screen coordinates (mouse click) to a specific Node on the game board.

    This function maps the 2D terminal character coordinates (y, x) to the
    logical (row, col) coordinates of the hexagonal grid.

    Rendering Logic Dependency:
    - Each node row takes up 3 lines of height (Node + Connection + Diagonal).
    - Even/Odd rows have different horizontal offsets.

    Args:
        board (System): The game board system model.
        my (int): The y-coordinate (row) of the mouse event.
        mx (int): The x-coordinate (column) of the mouse event.

    Returns:
        Node: The node at the clicked location, or None if the click was invalid/off-grid.
    """
    # The grid rendering uses 3 lines of vertical space per node row.
    # Line 0: Node
    # Line 1: Diagonal Connection
    # Line 2: Diagonal Connection
    if my % 3 != 0:
        return None
    
    row = my // 3

    # Calculate column based on row parity (hexagonal stagger)
    if row % 2 == 0:
        # Even Row: Nodes are at 0, 6, 12... (visual X coords approx)
        # Logic: mx // 3 gives "blocks", every other block is a node.
        col = mx // 3
        if col % 2 == 0:
            col = col // 2
            return board.node_at(row, col)
    else:
        # Odd Row: Indented by 3 chars.
        mx -= 3
        col = mx // 3
        if col % 2 == 0:
            col = col // 2
            return board.node_at(row, col)
    return None
