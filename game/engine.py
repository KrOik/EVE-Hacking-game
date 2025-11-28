"""
This module contains the core game engine for the EVE Hacking game.

It defines the GameEngine class, which acts as the central controller for the game.
It manages the main game loop, processes user input (keyboard and mouse), and coordinates
game state updates between the System (model) and SystemRenderer (view).

Architecture Role:
    - Controller: Mediates between the Model (System) and View (SystemRenderer).
    - Game Loop: Manages the frame-by-frame execution and input handling.

Modification History:
    - 2025-11-28: Created and moved game loop logic from main.py.
"""

import curses
import time
from .models import System
from .rendering import SystemRenderer
from .utils import get_node_at_mouse

class GameEngine(object):
    """
    Manages the main game loop and state of the hacking minigame.

    This class is responsible for:
    1. Initializing the game environment.
    2. Running the main loop (render -> input -> update).
    3. Handling user inputs (mouse clicks, keyboard commands).
    4. Detecting game over conditions.
    """

    def __init__(self, screen):
        """
        Initializes the game engine with the given curses screen.

        Args:
            screen (curses.window): The main curses screen object.
        """
        self.screen = screen
        # Initialize the game board (Model)
        self.board = System()
        # Initialize the renderer (View)
        self.renderer = SystemRenderer()
        # Configure curses settings
        self._setup_curses()

    def _setup_curses(self):
        """
        Configures the initial curses environment settings.
        
        Settings applied:
        - noecho: Don't print key presses to screen.
        - cbreak: React to keys instantly without waiting for Enter.
        - curs_set(0): Hide the cursor.
        - mousemask: Enable mouse event capture.
        """
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        # Resize terminal to ensure enough space (optional/platform dependent)
        curses.resize_term(128, 128)
        curses.start_color()
        self.screen.keypad(True) # Enable special keys (arrows, etc.)
        curses.mousemask(True)   # Enable all mouse events

    def run(self):
        """
        Starts and runs the main game loop.

        This is a blocking call that runs until the game ends or the user exits.
        
        The loop structure is:
        1. Render the current state.
        2. Check for Game Over (Virus death).
        3. Wait for and process user input.
        4. Update game state based on input.
        """
        while True:
            # --- Render Phase ---
            self.renderer.render(self.board, self.screen)

            # --- Game Over Check ---
            if self.board.virus.is_dead:
                # Display GAME OVER message at the bottom
                self.screen.addstr(self.board.height * 4, 0, "GAME OVER", curses.color_pair(1))
                self.screen.refresh()
                time.sleep(3)
                break

            # --- Input Phase ---
            ch = self.screen.getch()

            # --- Input Handling ---
            if ch == curses.KEY_MOUSE:
                # Handle Mouse Input
                _, mx, my, _, _ = curses.getmouse()
                node = get_node_at_mouse(self.board, my, mx)
                
                # Check if click is within the utility belt area
                if my == self.board.utility_belt_y:
                    # Calculate which utility slot was clicked
                    # Each slot is approx 4 chars wide, starting from x=11
                    utility_index = (mx - 11) // 4
                    if 0 <= utility_index < len(self.board.virus.utilities):
                        utility = self.board.virus.utilities[utility_index]
                        
                        # If utility has no target (e.g., SelfRepair, Shield), use immediately
                        if not hasattr(utility, 'target'):
                            self.board.virus.use_utility(utility)
                            self.board.end_turn()
                        else:
                            # Otherwise, set as pending to wait for target selection
                            self.board.pending_utility = utility
                
                elif self.board.pending_utility:
                    # If a utility is pending, this click is for selecting a target
                    if node and node.token:
                        self.board.virus.use_utility(self.board.pending_utility, node)
                        self.board.pending_utility = None
                        self.board.end_turn()
                
                elif self.board.can_visit_node(node):
                    # Standard move/attack action
                    self.board.visit_node(node)
                    self.board.end_turn()
            
            # Handle Restart (Enter key)
            if ch == curses.KEY_ENTER or ch == 10 or ch == 13:
                # Reset the game by creating a new System instance
                self.board = System()
