"""
This module handles the rendering of the system map on the terminal using the curses library.

It defines the SystemRenderer class, which is responsible for drawing the nodes,
connections, and status information of the system. It translates the abstract game
state (System) into visual characters and colors on the terminal screen.

Architecture Role:
    - View: Displays the game state to the user.
    - UI Rendering: Manages layout, colors, and ASCII art generation for the hexagonal grid.

Modification History:
    - 2025-11-28: Migrated from render.py to game/rendering.py and updated imports.
"""

import curses
from .models import Node, Core, Firewall


class SystemRenderer(object):
    """
    The SystemRenderer class is responsible for rendering the system map in the terminal.

    It uses the curses library to draw the game state, including the grid of nodes,
    the connections between them, and other status information like virus and core health.
    """

    def __init__(self):
        """
        Initializes the SystemRenderer.

        This constructor currently does not perform any specific initialization,
        but it can be extended to set up rendering-related configurations.
        """
        pass

    def get_node_string(self, system, node: Node) -> str:
        """
        Determines the string representation of a node based on its state.

        This method returns a 3-character string that visually represents the node's
        current status (e.g., visited, exposed, blocked, core, firewall).

        Args:
            system (System): The main System object containing the game state.
            node (Node): The Node object to be rendered.

        Returns:
            str: A 3-character string representing the node's appearance.
        """
        # 🔧 repair
        #  secondary vector
        # 🛡️ - shield
        # 💔 - kernel rot
        if node is None:
            # Return an empty string for non-existent nodes.
            return '   '

        # Check for active effects on this node's token (e.g., Secondary Vector)
        if node.token:
            for effect in system.virus.active_effects:
                if getattr(effect, 'target', None) == node.token:
                    return f'{effect.icon} '

        if node.is_blocked:
            # 'x' indicates that the node is blocked.
            return f' x '
        elif (node.is_visited or system.debug_mode) and isinstance(node.token, Core):
            # A computer icon represents a visited core node.
            return f'🖥️'
        elif (node.is_visited or system.debug_mode) and node.token and node.token.icon:
            return f'{node.token.icon} '
        elif system.selected_node == node:
            # If the node is selected, show the distance to the nearest meaningful node.
            # This acts as a "hot/cold" hint for finding hidden items/cores.
            dist = system.get_nearest_meaningful_node_distance(node)
            return '(' + str(min(dist, 5)) + ')'
        elif node.is_visited:
            # Parentheses indicate a visited node.
            # If we want the hint to persist on visited nodes (memory aid), we could show it here too.
            # But standard EVE behavior often clears it or keeps it if it was a "safe" node.
            # For now, let's just show empty visited status unless it's selected.
            return '( )'
        elif node.is_exposed:
            # Brackets indicate an exposed but not yet visited node.
            # If the node has a utility/token, show it? (Usually hidden until visited)
            # User requested "icon displayed on map (on item node)". 
            # Assuming they want visibility when exposed:
            if node.token and isinstance(node.token, (Core, Firewall)):
                 # Keep Core/Firewall hidden until visited as per standard mechanics? 
                 # Or reveal all if exposed?
                 # Standard EVE: revealed when adjacent visited.
                 pass
            return f'[ ]'
        else:
            # A bullet point represents a hidden/unexposed node.
            return ' • '

    def render(self, system, screen):
        """
        Renders the entire system map on the provided curses screen.

        This is the main rendering method. It iterates through the system's nodes,
        draws them and their connections, and displays status information.
        
        Rendering Logic:
        The grid is hexagonal, but stored in a 2D array. To render this:
        1. Odd rows are indented by spaces.
        2. Connections are drawn between horizontal neighbors.
        3. Diagonal connections are drawn between rows, handling the even/odd row offset logic.

        Args:
            system (System): The System object representing the current game state.
            screen (curses.window): The curses screen object to draw on.
        """
        # Initialize color pairs for rendering (e.g., for text colors).
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        y = 0  # Initialize the y-coordinate for drawing.

        # Iterate over each row of nodes in the system's grid.
        for row_index, row in enumerate(system.nodes):
            s = ''  # This string will hold the content of the current row of nodes.
            if row_index % 2 == 1:
                # Indent odd rows to create a hexagonal grid layout.
                s += '   '

            # Draw the nodes in the current row.
            for column_index in range(system.width):
                lhs = system.nodes[row_index][column_index]
                rhs = None if column_index >= system.width - 1 else system.nodes[row_index][column_index + 1]
                s += self.get_node_string(system, lhs)
                # Draw a horizontal connection if two adjacent nodes exist.
                if lhs is not None and rhs is not None:
                    s += '---'
                else:
                    s += '   '

            # Add the rendered row of nodes to the screen.
            screen.addstr(y, 0, s)
            y += 1

            # Skip drawing edges for the last row.
            if row_index >= system.height - 1:
                continue

            # These strings will hold the diagonal connections between rows.
            c = ' '
            d = ' '

            # --- Hexagonal Connection Logic ---
            # The logic for drawing diagonal connections depends on whether the row is even or odd.
            # Even Rows (0, 2, 4...):
            #   Nodes align directly above the "gaps" of the odd row below.
            #   Connections go Down-Right (\) and Down-Left (/).
            # Odd Rows (1, 3, 5...):
            #   Nodes are indented.
            #   Connections go Down-Left (/) and Down-Right (\).
            if row_index % 2 == 0:
                # --- Drawing connections for even rows ---
                for column_index in range(system.width):
                    lhs = system.nodes[row_index][column_index]

                    if lhs is not None:
                        # Check for a connection to the node directly below.
                        # In even rows, 'down' connects to the bottom-left neighbor visually?
                        # Actually, due to indentation:
                        # Even Row i, Col j connects to:
                        #   Row i+1, Col j-1 (South-West)
                        #   Row i+1, Col j   (South-East)
                        if system.nodes[row_index + 1][column_index] is not None:
                            c += r' \  '
                            d += r'  \ '
                        else:
                            c += '    '
                            d += '    '
                    else:
                        c += '    '
                        d += '    '

                    if column_index + 1 >= system.width:
                        continue

                    # Check for a connection to the node diagonally to the right.
                    rhs = system.nodes[row_index][column_index + 1]
                    if rhs is not None and 0 <= column_index < system.width - 1 and system.nodes[row_index + 1][column_index] is not None:
                        c += r' /'
                        d += r'/ '
                    else:
                        c += '  '
                        d += '  '
            else:
                # --- Drawing connections for odd rows ---
                for column_index in range(system.width):
                    lhs = system.nodes[row_index][column_index]

                    if lhs is not None:
                        # Check for a connection to the node diagonally to the left on the row below.
                        if system.nodes[row_index + 1][column_index] is not None:
                            c += r'  / '
                            d += r' /  '
                        else:
                            c += '    '
                            d += '    '
                    else:
                        c += '    '
                        d += '    '

                    if column_index + 1 >= system.width:
                        continue
                    
                    # Check for a connection to the node diagonally to the right on the row below.
                    if column_index < system.width - 1 and \
                            lhs is not None and \
                            system.nodes[row_index + 1][column_index + 1] is not None:
                        c += r'\ '
                        d += ' \\'
                    else:
                        c += '  '
                        d += '  '

            # Add the rendered connection lines to the screen.
            screen.addstr(y, 0, c)
            y += 1
            screen.addstr(y, 0, d)
            y += 1

        y += 1

        # Display the status of the virus.
        # TODO: Optimization - Iterate over all the exposed tokens on the board and print them out separately if needed.
        if not system.virus.is_dead:
            virus_status = f'VIRUS: ({system.virus.coherence}/{system.virus.strength})'
            
            # Add active effects status
            active_effects = []
            if system.virus.shield_charges > 0:
                active_effects.append(f'Shields: {system.virus.shield_charges} 💠')
            
            for effect in system.virus.active_effects:
                if getattr(effect, 'name', '') == 'Self-Repair':
                    active_effects.append(f'Repairing ({effect.duration}) 🩹')
            
            if active_effects:
                virus_status += ' | ' + ' | '.join(active_effects)

            screen.addstr(y, 0, virus_status)
            y += 1

        # --- Render utility belt ---
        utility_belt_str = "Utilities: "
        for i in range(3):
            if i < len(system.virus.utilities):
                utility = system.virus.utilities[i]
                utility_belt_str += f'[{utility.icon}] '
            else:
                utility_belt_str += '[ ] '
        screen.addstr(y, 0, utility_belt_str)
        system.utility_belt_y = y
        y += 1

        if system.pending_utility:
            screen.addstr(y, 0, f'TARGETING: Select a target for {system.pending_utility.name}')
            y += 1

        # Display the status of the core if it has been visited.
        if system.core.node.is_visited:
            screen.addstr(y, 0, f'CORE ({system.core.coherence}/{system.core.strength})')
            y += 1

        # Refresh the screen to show all the updates.
        screen.refresh()
