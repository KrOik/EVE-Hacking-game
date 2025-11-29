import time
import sys
import os
import argparse
import curses
import multiprocessing

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.models import System
from game.autopilot import AutoPilot, Config
from game.rendering import SystemRenderer

def run_single_game(params=None):
    """Runs a single game silently and returns stats."""
    if params:
        Config.update(params)
        
    system = System()
    pilot = AutoPilot(system)
    
    turns = 0
    max_turns = 1000 # Safety limit to prevent infinite loops
    while not system.virus.is_dead and not system.core.is_dead:
        turns += 1
        if turns > max_turns:
            break
            
        action_taken = pilot.step()
        if not action_taken:
            break
            
    is_win = system.core.is_dead
    return {
        'result': 'win' if is_win else 'loss',
        'turns': turns,
        'health': system.virus.coherence
    }

def run_visual_game(stdscr, params=None, delay=0.3, debug_mode=False):
    """Runs a single game with visualization."""
    if params:
        Config.update(params)
    
    # Setup curses similar to GameEngine
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    
    # Check terminal size
    max_y, max_x = stdscr.getmaxyx()
    # Board uses approx (height * 3) + 5 lines. For height=8, that's ~29 lines.
    # Let's require at least 30 lines to be safe, or warn.
    
    # Initialize game
    system = System(debug_mode=debug_mode)
    pilot = AutoPilot(system)
    renderer = SystemRenderer()
    
    while not system.virus.is_dead and not system.core.is_dead:
        # Render
        try:
            renderer.render(system, stdscr)
        except curses.error:
            # If rendering the board fails, the terminal is definitely too small
            raise Exception(f"Terminal too small for board rendering (Size: {max_y}x{max_x})")
        
        # Display AI Status
        # Calculate dynamic Y position based on board size
        # Board bottom is roughly: (system.height * 3) + 5 (status lines)
        status_y = (system.height * 3) + 6
        
        # Safety check for bounds
        if status_y >= max_y:
            status_y = max_y - 1
            
        try:
            stdscr.addstr(status_y, 0, f"AutoPilot Action: {pilot.last_action} (Delay: {delay}s)")
            stdscr.clrtoeol() # Clear rest of line
            stdscr.refresh()
        except curses.error:
             pass # Ignore status line errors if out of bounds
        
        time.sleep(delay)
        
        # Step
        action_taken = pilot.step()
        
        if not action_taken:
            try:
                if status_y + 1 < max_y:
                    stdscr.addstr(status_y + 1, 0, "AI Stuck! No valid moves.")
                    stdscr.refresh()
            except curses.error:
                pass
            time.sleep(2)
            break
            
    # Final State
    try:
        renderer.render(system, stdscr)
        result = "VICTORY" if system.core.is_dead else "FAILURE"
        
        game_over_y = (system.height * 3) + 8
        if game_over_y >= max_y - 1:
            game_over_y = max_y - 2
            
        stdscr.addstr(game_over_y, 0, f"GAME OVER: {result}", curses.A_BOLD)
        stdscr.addstr(game_over_y + 1, 0, "Press any key to exit...")
        stdscr.refresh()
        stdscr.getch()
    except curses.error:
        pass

def run_batch_simulation(num_games=100, params=None):
    """Runs a batch of games and prints statistics using multiprocessing."""
    cpu_count = multiprocessing.cpu_count()
    print(f"Running simulation for {num_games} games using {cpu_count} processes...")
    if params:
        print(f"Custom Params: {params}")
        
    start_time = time.time()
    wins = 0
    total_turns = 0
    total_health = 0
    
    try:
        # We use a timeout for the map result to prevent hanging forever
        # But imap_unordered doesn't support timeout directly on the iterator in a way that kills stuck tasks easily.
        # Instead, we'll use apply_async or just rely on the fact that we have logic to prevent infinite loops.
        # To be safe against infinite loops in step(), let's add a max_turns limit in run_single_game.
        
        with multiprocessing.Pool() as pool:
            # Use imap_unordered to get results as they finish for the progress bar
            results = pool.imap_unordered(run_single_game, [params] * num_games)
            
            for i, stats in enumerate(results):
                if stats['result'] == 'win':
                    wins += 1
                    total_health += stats['health']
                total_turns += stats['turns']
                
                # Progress bar
                progress = (i + 1) / num_games
                bar_length = 40
                block = int(round(bar_length * progress))
                text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {i+1}/{num_games}"
                sys.stdout.write(text)
                sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted!")
        # Pool will be closed automatically by context manager, but we might want to terminate explicitly if stuck
        # In a simple script, exiting usually kills children.
            
    print("\n")
    elapsed = time.time() - start_time
    
    win_rate = (wins / num_games) * 100 if num_games > 0 else 0
    avg_turns = total_turns / num_games if num_games > 0 else 0
    avg_health = total_health / wins if wins > 0 else 0
    
    print(f"Simulation Complete in {elapsed:.2f}s")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Turns: {avg_turns:.2f}")
    print(f"Avg Health (Wins): {avg_health:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVE Hacking AI Simulation")
    parser.add_argument('--visual', '-v', action='store_true', help='Run in visual mode (curses)')
    parser.add_argument('--games', '-n', type=int, default=100, help='Number of games for batch simulation')
    parser.add_argument('--delay', '-d', type=float, default=0.2, help='Delay between steps in visual mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (reveal all nodes)')
    
    args = parser.parse_args()
    
    if args.visual:
        try:
            curses.wrapper(run_visual_game, params=None, delay=args.delay, debug_mode=args.debug)
        except Exception as e:
            print(f"Error in visual mode: {e}")
    else:
        run_batch_simulation(num_games=args.games)
