"""
This module defines the core data models for the EVE Hacking game.

It serves as the "Model" in the MVC architecture, containing all the game logic,
state management, and data structures. It includes classes for the game grid (System),
individual nodes (Node), and the various entities (Tokens) that populate the grid,
such as the Virus, Core, Firewalls, and Utilities.

Architecture Role:
    - Model: Encapsulates application state and business logic.
    - State Management: Handles board generation, turn processing, and entity interactions.

Modification History:
    - 2025-11-28: Migrated from data.py to game/models.py and added comprehensive docstrings.
"""

import random
import curses
from collections import deque
import time
import os
from game.map_generator import MapGenerator

# --- Direction Constants ---
# These bitmasks represent the six possible directions in a hexagonal grid.
WEST = 1
NORTH_WEST = 2
NORTH_EAST = 4
EAST = 8
SOUTH_EAST = 16
SOUTH_WEST = 32
ALL = (WEST | NORTH_WEST | NORTH_EAST | EAST | SOUTH_EAST | SOUTH_WEST)


class SnowflakeIDGenerator:
    """
    A distributed unique ID generator based on Twitter's Snowflake algorithm.

    This class generates unique 64-bit integers used for seeding the random number generator
    for map generation. This allows for reproducible maps sharing the same seed.

    Attributes:
        epoch (int): Custom epoch timestamp in milliseconds. Default is 1640995200000.
        machine_id (int): Unique ID of the machine/process (0-1023).
        sequence (int): Sequence number for IDs generated in the same millisecond.
        last_timestamp (int): The timestamp of the last generated ID to handle clock drifts/collisions.
    """
    def __init__(self, epoch: int = 1640995200000):
        self.machine_id = int(os.getenv("MACHINE_ID", "0")) & 0x3FF
        self.epoch = epoch
        self.sequence = 0
        self.last_timestamp = -1

    def _current_timestamp(self):
        """Returns the current system timestamp in milliseconds."""
        return int(time.time() * 1000)

    def generate_id(self):
        """
        Generates a unique Snowflake ID.

        Returns:
            int: A unique 64-bit integer composed of timestamp, machine ID, and sequence.
        """
        timestamp = self._current_timestamp()

        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & 0xFFF
            if self.sequence == 0:
                # Sequence exhausted, wait for next millisecond
                while timestamp <= self.last_timestamp:
                    timestamp = self._current_timestamp()
        else:
            self.sequence = 0

        self.last_timestamp = timestamp

        return (
            ((timestamp - self.epoch) << 22) |
            (self.machine_id << 12) |
            self.sequence
        )


class Node(object):
    """
    Represents a single hexagonal node in the system grid.

    Nodes are the fundamental units of the game map. They can hold Tokens (entities),
    have states (visited, exposed, blocked), and are connected to up to 6 neighbors.

    Attributes:
        row (int): The row index in the grid.
        column (int): The column index in the grid.
        is_visited (bool): Whether the player has successfully entered this node.
        is_exposed (bool): Whether the node is visible/accessible to the player.
        block_count (int): Number of active effects preventing entry to this node.
        input (int): Reserved for future mechanics (e.g., keyboard shortcuts).
        num_neighbors (int): Cached count of valid neighbors.
        token (Token): The entity occupying this node, or None.
    """
    def __init__(self, row: int, column: int):
        self.row = row
        self.column = column
        self.is_visited = False
        self.is_exposed = False
        self.block_count = 0
        self.input = 0
        self.num_neighbors = 0
        self.__token = None

    @property
    def is_blocked(self):
        """Returns True if the node is currently blocked by any effect (block_count > 0)."""
        return self.block_count > 0

    def on_exposed(self):
        """
        Callback triggered when the node becomes exposed (visible).
        
        This is typically called when an adjacent node is visited.
        """
        pass

    def on_attacked(self):
        """
        Callback triggered when the player attempts to enter/attack this node.
        """
        pass

    @property
    def token(self):
        """The Token object placed on this node. Automatically manages bidirectional reference."""
        return self.__token

    @token.setter
    def token(self, x):
        if self.__token is not None:
            self.__token.node = None
        self.__token = x
        if self.__token is not None:
            self.__token.node = self


class BfsNode(object):
    """
    A helper class for Breadth-First Search (BFS) operations.
    
    Used to track the path reconstruction and depth during graph traversal.
    
    Attributes:
        node (Node): The current graph node.
        parent (BfsNode): The previous node in the path (for backtracking).
        depth (int): The distance from the start node.
    """
    def __init__(self, node, parent, depth):
        self.node = node
        self.parent = parent
        self.depth = depth


class System(object):
    """
    Represents the entire game system/board.

    This is the central class that manages:
    1. Procedural Map Generation (Nodes, Connections, Token placement).
    2. Game State (Virus status, Core status, Turn management).
    3. Navigation and Pathfinding (BFS, Neighbor lookup).
    4. Interaction Logic (Visiting nodes, Combat resolution).

    Attributes:
        width (int): Grid width (fixed at 14).
        height (int): Grid height (fixed at 8).
        nodes (list[list[Node]]): 2D array of Node objects.
        virus (Virus): The player's character.
        core (Core): The main objective.
        selected_node (Node): The currently highlighted/selected node.
    """
    def __init__(self, seed=None, debug_mode=False):
        """
        Initializes the System and generates a new game map.

        Args:
            seed (int, optional): Seed for RNG to create deterministic maps. 
                                  If None, a new Snowflake ID is used.
            debug_mode (bool): If True, all nodes are revealed (exposed) initially.
        """
        if seed is None:
            id_generator = SnowflakeIDGenerator()
            seed = id_generator.generate_id()
        self.debug_mode = debug_mode
        self.width = 14 # Fixed grid width
        self.height = 8 # Fixed grid height
        self.nodes = []
        self.selected_node = None
        self.core = None
        self.exposed_count = 0
        self.virus = Virus(self)
        self.utility_belt_y = -1
        self.pending_utility = None

        # Start the procedural generation process
        self.create_nodes(seed)

        # Debug Mode: Reveal all nodes
        if debug_mode:
            for row in self.nodes:
                for node in row:
                    if node:
                        node.is_exposed = True

    def bfs_iterator(self, node):
        """
        A generator that yields nodes in Breadth-First Search order.

        Args:
            node (Node): The starting node.

        Yields:
            BfsNode: A wrapper containing the current node, its parent, and depth.
        """
        if node is None:
            raise ValueError
        queue = deque()
        queue.append(BfsNode(node, None, 0))
        visited = {node}
        while len(queue) > 0:
            cursor = queue.pop()
            yield cursor
            for neighbor in self.get_neighbors(cursor.node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.insert(0, BfsNode(neighbor, cursor, cursor.depth + 1))

    def prune_disjoint(self, node):
        """
        Removes all nodes that are not reachable from the given start node.
        
        This ensures the map is a single connected component.

        Args:
            node (Node): A node known to be in the main component (e.g., start node).
        """
        nodes = set([x.node for x in self.bfs_iterator(node)])
        for row in range(self.height):
            for col in range(self.width):
                node = self.nodes[row][col]
                if node not in nodes:
                    self.nodes[row][col] = None

    def create_nodes(self, seed):
        """
        Procedurally generates the game map using MapGenerator.

        Args:
            seed (int): Random seed for generation.
        """
        MapGenerator(self).generate(seed)

    def remove_node(self, node):
        """Removes a node from the grid and updates neighbor counts."""
        for neighbor in self.get_neighbors(node):
            neighbor.num_neighbors -= 1
        self.nodes[node.row][node.column] = None

    def is_valid_index(self, row, col):
        """Checks if (row, col) is within grid boundaries."""
        return 0 <= row < self.height and 0 <= col < self.width

    def can_visit_node(self, node):
        """
        Determines if the player can legally interact with a node.

        Rules:
        - Cannot visit None.
        - Cannot visit already visited nodes (unless they have a token to interact with).
        - Cannot visit blocked nodes.
        - Must be exposed (adjacent to a visited node).

        Args:
            node (Node): The target node.

        Returns:
            bool: True if visitable.
        """
        if node is None:
            return False
        if (node.is_visited and node.token is None) or node.is_blocked:
            return False
        return node.is_exposed

    def visit_node(self, node, force=False):
        """
        Executes the logic for visiting/interacting with a node.

        This includes:
        1. Marking as visited/exposed.
        2. Exposing neighbors.
        3. Triggering Token interactions (Attack or Collect).

        Args:
            node (Node): The node to visit.
            force (bool): Bypass validation checks (used for start node).
        """
        if not force and not self.can_visit_node(node):
            return

        if node is None:
            raise RuntimeError('cannot visit a null node!')

        if not node.is_visited:
            # First visit: Reveal the node and surroundings
            node.is_visited = True
            node.is_exposed = True
            for neighbor in self.get_neighbors(node):
                if not neighbor.is_exposed:
                    neighbor.is_exposed = True
            if node.token is not None:
                node.token.on_exposed()
        else:
            # Subsequent visit: Interaction (Attack or Collect)
            if node.token is not None:
                if isinstance(node.token, Utility):
                    # Collect Utility
                    if self.virus.add_utility(node.token):
                        node.token = None
                else:
                    # Attack Defensive Token
                    node.token.on_attacked(self.virus)

        self.selected_node = node

    def get_starting_node(self):
        """
        Finds an ideal starting node, preferring edges with fewer neighbors.
        """
        m = 100
        starting_nodes = []
        for row in range(self.height):
            for column in range(self.width):
                node = self.nodes[row][column]
                if node is None:
                    continue
                num_neighbors = self.get_num_neighbors(node)
                if num_neighbors < m:
                    m = num_neighbors
                    starting_nodes = [node]
                elif num_neighbors == m:
                    starting_nodes.append(node)
        return random.choice(starting_nodes)

    def get_nodes_at_jumps(self, node, jumps):
        """Returns all nodes exactly `jumps` steps away via BFS."""
        queue = deque()
        queue.append(BfsNode(node, None, 0))
        nodes = []
        visited = {node}
        while len(queue) > 0:
            cursor = queue.pop()
            neighbors = self.get_neighbors(cursor.node)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                if cursor.depth >= jumps:
                    nodes.append(neighbor)
                    break
                new = BfsNode(neighbor, cursor, cursor.depth + 1)
                queue.insert(0, new)
        return nodes

    def get_neighbor(self, node, direction):
        """
        Calculates the neighbor of a node in a hexagonal grid.

        Hexagonal Grid Logic:
        - Rows are offset. Odd rows are indented.
        - Directions depend on row parity (even/odd).
        
        Args:
            node (Node): Source node.
            direction (int): Direction constant (e.g., NORTH_EAST).

        Returns:
            Node: The neighbor or None.
        """
        if node is None:
            raise RuntimeError()

        row, column = node.row, node.column

        # Horizontal neighbors are simple
        if direction == WEST:
            if column <= 0:
                return None
            return self.nodes[row][column - 1]
        elif direction == EAST:
            if column >= self.width - 1:
                return None
            return self.nodes[row][column + 1]

        # Diagonal neighbors depend on row parity
        if row % 2 == 0:  # Even row
            if direction == NORTH_WEST and row > 0 and column > 0:
                return self.nodes[row - 1][column - 1]
            elif direction == NORTH_EAST and row > 0:
                return self.nodes[row - 1][column]
            elif direction == SOUTH_WEST and row < self.height - 1 and column > 0:
                return self.nodes[row + 1][column - 1]
            elif direction == SOUTH_EAST and row < self.height - 1:
                return self.nodes[row + 1][column]
        else:  # Odd row (indented)
            if direction == NORTH_WEST and row > 0:
                return self.nodes[row - 1][column]
            elif direction == NORTH_EAST and row > 0 and column < self.width - 1:
                return self.nodes[row - 1][column + 1]
            elif direction == SOUTH_WEST and row < self.height - 1:
                return self.nodes[row + 1][column]
            elif direction == SOUTH_EAST and row < self.height - 1 and column < self.width - 1:
                return self.nodes[row + 1][column + 1]

        return None

    def get_neighbors(self, node):
        """Returns a list of all valid neighbors for a node."""
        if node is None:
            return []
        neighbors = []
        for direction in map(lambda x: 1 << x, range(6)):
            neighbor = self.get_neighbor(node, direction)
            if neighbor is not None:
                neighbors.append(neighbor)
        return neighbors

    def get_num_neighbors(self, node):
        """Returns the count of neighbors."""
        return sum(1 for _ in self.get_neighbors(node))

    def get_path(self, start: Node, destination: Node):
        """
        Finds shortest path between two nodes using BFS.
        
        Returns:
            list[Node]: Path from start to destination (exclusive of start?).
        """
        if start == destination:
            return []
        queue = deque()
        queue.append(BfsNode(start, None, 0))
        path = []
        visited = set()
        while len(queue) > 0:
            cursor = queue.pop()
            visited.add(cursor.node)
            if cursor.node == destination:
                while cursor.parent is not None:
                    path.insert(0, cursor.node)
                    cursor = cursor.parent
                return path
            neighbors = self.get_neighbors(cursor.node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    child = BfsNode(neighbor, cursor, cursor.depth)
                    queue.insert(0, child)
                    visited.add(neighbor)
        return None

    def get_nearest_meaningful_node_distance(self, start_node: Node) -> int:
        """
        Calculates BFS distance to the nearest 'meaningful' node (Node with Token).

        Used for the 'hot/cold' distance hint.
        """
        if start_node is None:
            return 0
        if start_node.token is not None:
            return 0

        queue = deque()
        queue.append(BfsNode(start_node, None, 0))
        visited = {start_node}
        
        while len(queue) > 0:
            cursor = queue.pop()
            
            if cursor.node.token is not None:
                return cursor.depth
                
            neighbors = self.get_neighbors(cursor.node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new = BfsNode(neighbor, cursor, cursor.depth + 1)
                    queue.insert(0, new)
        return 0

    def node_at(self, row, col):
        """Safe accessor for grid nodes."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.nodes[row][col]
        return None

    def get_node_for_input(self, char):
        """Deprecated: Gets node mapped to a specific character input."""
        char = char - 97
        for row in range(self.height):
            for column in range(self.width):
                node = self.nodes[row][column]
                if node is not None and not node.is_visited and node.is_exposed and node.input == char:
                    return node
        return None

    def end_turn(self):
        """
        Processes all end-of-turn effects.
        
        Includes:
        1. RestoNode healing logic.
        2. Active Virus Effect updates (SelfRepair, SecondaryVector).
        """
        # Process RestoNode healing
        for row in self.nodes:
            for node in row:
                if node and node.token and isinstance(node.token, RestoNode):
                    node.token.on_end_turn()

        # Process Virus effects
        for effect in self.virus.active_effects[:]:
            effect.on_end_turn()
            if effect.is_finished:
                self.virus.active_effects.remove(effect)


class Token(object):
    """
    Base class for all entities that occupy a node.

    Attributes:
        system (System): Reference to the game board.
        node (Node): The node this token occupies.
        coherence (int): Health/Integrity points.
        strength (int): Attack/Damage points.
        icon (str): Visual representation.
    """
    def __init__(self, system, node=None):
        self.node = node
        self.system = system
        self.coherence = 0
        self.strength = 0
        self.icon = ''

    def on_attacked(self, attacker):
        """
        Handles incoming attacks.
        
        Args:
            attacker (Token): The attacking entity (usually Virus).
        """
        self.take_damage(attacker.strength)
        if not self.is_dead:
            # Counter-attack logic
            if isinstance(attacker, Virus) and attacker.shield_charges > 0:
                attacker.shield_charges -= 1
            else:
                attacker.take_damage(self.strength)

    def take_damage(self, coherence):
        """Reduces coherence by the specified amount."""
        self.coherence = max(0, self.coherence - coherence)
        if self.is_dead:
            self.on_destroyed()

    @property
    def is_dead(self):
        return self.coherence <= 0

    def on_exposed(self):
        """Triggered when the token is revealed."""
        pass

    def on_destroyed(self):
        """Triggered when coherence reaches 0."""
        if self.node is not None:
            self.node.token = None


class Virus(Token):
    """
    The Player Character.
    
    Attributes:
        utilities (list): Inventory of collected utilities.
        shield_charges (int): Number of active shield layers.
        active_effects (list): Currently active buffs/debuffs.
    """
    def __init__(self, system, node=None):
        super().__init__(system, node)
        self.coherence = 115
        self.strength = 45
        self.utilities = []
        self.shield_charges = 0
        self.active_effects = []

    def add_utility(self, utility):
        """Collects a utility if inventory space permits (Max 3)."""
        if len(self.utilities) < 3:
            self.utilities.append(utility)
            return True
        return False

    def use_utility(self, utility, target=None):
        """Activates a utility effect."""
        if utility not in self.utilities:
            return

        if isinstance(utility, SelfRepair):
            self.active_effects.append(utility)
            self.utilities.remove(utility)

        elif isinstance(utility, KernelRot):
            if target and target.token:
                target.token.take_damage(999) # Instakill
                self.utilities.remove(utility)

        elif isinstance(utility, PolymorphicShield):
            self.shield_charges += 3
            self.utilities.remove(utility)

        elif isinstance(utility, SecondaryVector):
            if target and target.token:
                utility.target = target.token
                self.active_effects.append(utility)
                self.utilities.remove(utility)

    def take_damage(self, coherence):
        """Takes damage, consuming shields if available."""
        if self.shield_charges > 0:
            self.shield_charges -= 1
        else:
            super().take_damage(coherence)

    def on_destroyed(self):
        pass


class Core(Token):
    """The Main Objective. Destroying this typically wins the game (not yet implemented)."""
    def __init__(self, system):
        super().__init__(system)
        self.coherence = 90
        self.strength = 10

    def on_destroyed(self):
        pass


class DataCache(Token):
    """
    A storage node that reveals its contents when accessed.
    Contains either a Utility or a Defensive Subsystem.
    """
    def __init__(self, system, inner_token):
        super().__init__(system)
        self.inner_token = inner_token
        self.icon = '📦'
        self.coherence = 1
        self.strength = 0

    def on_attacked(self, attacker):
        """Reveals the hidden token when attacked/interacted with."""
        # Swap immediately
        if self.node:
            node = self.node
            node.token = self.inner_token
            self.inner_token.on_exposed()


class Firewall(Token):
    """
    A defensive node that buffs neighbors.
    
    Effect: Blocks entry to all adjacent nodes while active.
    """
    def __init__(self, system):
        super().__init__(system)
        self.coherence = 90
        self.strength = 20
        self.icon = '🔥'

    @property
    def can_be_attacked(self):
        return True

    def on_exposed(self):
        """Blocks all neighbors when revealed."""
        super().on_exposed()
        for neighbor in self.system.get_neighbors(self.node):
            neighbor.block_count += 1

    def on_destroyed(self):
        """Unblocks neighbors when destroyed."""
        for neighbor in self.system.get_neighbors(self.node):
            neighbor.block_count -= 1
        super().on_destroyed()


class Utility(Token):
    """Abstract base class for pickup items."""
    def __init__(self, system):
        super().__init__(system)
        self.icon = ''
        self.name = ''
        self.description = ''


class SelfRepair(Utility):
    """Heals the Virus over 3 turns."""
    def __init__(self, system):
        super().__init__(system)
        self.icon = '🩹'
        self.name = 'Self-Repair'
        self.description = 'Increases Virus Coherence by 5-10 each turn for 3 turns.'
        self.duration = 3
        self.is_finished = False

    def on_end_turn(self):
        if self.duration > 0:
            self.system.virus.coherence += random.randint(5, 10)
            self.duration -= 1
        if self.duration <= 0:
            self.is_finished = True


class KernelRot(Utility):
    """Instantly destroys a target node (Massive damage)."""
    def __init__(self, system):
        super().__init__(system)
        self.icon = '💔'
        self.name = 'Kernel Rot'
        self.description = 'Reduces a System Coherence by 50%.'
        self.target = None


class PolymorphicShield(Utility):
    """Grants 3 charges of invulnerability."""
    def __init__(self, system):
        super().__init__(system)
        self.icon = '💠'
        self.name = 'Polymorphic Shield'
        self.description = 'Nullifies the next two System attacks against your Virus.'


class SecondaryVector(Utility):
    """DoT (Damage over Time) attack on a target node."""
    def __init__(self, system):
        super().__init__(system)
        self.icon = '🎯'
        self.name = 'Secondary Vector'
        self.description = 'Reduces a System Coherence by 20 each turn for three turns.'
        self.duration = 3
        self.is_finished = False
        self.target = None

    def on_end_turn(self):
        if self.duration > 0 and self.target and not self.target.is_dead:
            self.target.coherence -= 20
            self.duration -= 1
        if self.duration <= 0 or not self.target or self.target.is_dead:
            self.is_finished = True


class AntiVirus(Token):
    """
    High Damage defensive node. Blocks neighbors.
    """
    def __init__(self, system):
        super().__init__(system)
        self.coherence = 60
        self.strength = 40
        self.icon = '🛡️'

    def on_exposed(self):
        super().on_exposed()
        for neighbor in self.system.get_neighbors(self.node):
            neighbor.block_count += 1

    def on_destroyed(self):
        for neighbor in self.system.get_neighbors(self.node):
            neighbor.block_count -= 1
        super().on_destroyed()


class RestoNode(Token):
    """
    Support node. Heals other defensive nodes each turn. Blocks neighbors.
    """
    def __init__(self, system):
        super().__init__(system)
        self.coherence = 80
        self.strength = 10
        self.icon = '⚕️'

    def on_exposed(self):
        super().on_exposed()
        for neighbor in self.system.get_neighbors(self.node):
            neighbor.block_count += 1

    def on_destroyed(self):
        for neighbor in self.system.get_neighbors(self.node):
            neighbor.block_count -= 1
        super().on_destroyed()

    def on_end_turn(self):
        """Heals a random damaged defensive subsystem."""
        defensive_subsystems = []
        for row in self.system.nodes:
            for node in row:
                if node and node.token and isinstance(node.token, (Firewall, AntiVirus, Suppressor, Core)):
                    defensive_subsystems.append(node.token)
        
        if defensive_subsystems:
            target = random.choice(defensive_subsystems)
            target.coherence = min(target.coherence + 20, 100)


class Suppressor(Token):
    """
    Debuff node. Reduces Virus attack strength. Blocks neighbors.
    """
    def __init__(self, system):
        super().__init__(system)
        self.coherence = 60
        self.strength = 15
        self.icon = '⬇️'

    def on_exposed(self):
        super().on_exposed()
        # Reduce virus strength
        self.system.virus.strength = max(10, self.system.virus.strength - 15)
        for neighbor in self.system.get_neighbors(self.node):
            neighbor.block_count += 1

    def on_destroyed(self):
        # Restore virus strength
        self.system.virus.strength += 15
        for neighbor in self.system.get_neighbors(self.node):
            neighbor.block_count -= 1
        super().on_destroyed()
