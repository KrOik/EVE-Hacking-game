"""
This module serves as the main entry point for the EVE Hacking game.

It handles the initialization of the curses environment and delegates the game loop
execution to the GameEngine. This separation of concerns ensures that the entry point
remains lightweight and focused on environment setup.

Architecture Role:
    - Entry Point: Bootstraps the application.
    - Environment Setup: Uses curses.wrapper to safely initialize/teardown the terminal.

Modification History:
    - 2025-11-28: Refactored to be a simple entry point.
"""

import curses
from game.engine import GameEngine
from game.logger import setup_logging
import logging

def main(screen):
    """
    The main function that initializes and runs the game.

    This function is called by curses.wrapper() and serves as the bridge between
    the curses environment and the game engine.

    Args:
        screen (curses.window): The main curses screen object provided by curses.wrapper.
                                This object represents the terminal window.

    Returns:
        None

    Raises:
        Exception: Propagates any unhandled exceptions from the game engine, ensuring
                   curses is torn down correctly before crashing.
    """
    # Initialize logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting EVE Hacking Game...")

    # Initialize the GameEngine with the prepared curses screen
    try:
        engine = GameEngine(screen)
        
        # Start the main game loop
        engine.run()
    except Exception as e:
        logger.exception("An unhandled exception occurred during game execution:")
        raise e
    finally:
        logger.info("Game shutting down.")

if __name__ == '__main__':
    # The curses.wrapper function is a safe way to run a curses application.
    # It handles the initialization and teardown of the curses environment,
    # ensuring that the terminal is restored to its normal state even if the program crashes.
    # This prevents the terminal from being left in an unusable state.
    curses.wrapper(main)
