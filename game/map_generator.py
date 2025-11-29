import random
from collections import deque
import math

class MapGenerator:
    """
    Encapsulates the procedural generation logic for the game map.
    
    This class is responsible for creating the grid of nodes, setting up the
    initial topology (start node, core, waypoints), creating the "maze" structure
    via random deletions, and populating the map with various tokens (Firewalls,
    Utilities, etc.) based on strategic rules.
    """
    def __init__(self, system):
        """
        Initializes the MapGenerator.

        Args:
            system (System): The game system instance that will hold the generated map.
        """
        self.system = system
        
    def generate(self, seed):
        """
        Procedurally generates the complete game map.

        The generation process follows these main steps:
        1. Initialize the grid with empty Nodes.
        2. Select a starting node on the edge.
        3. Determine the location of the System Core and a Waypoint.
        4. Lock a guaranteed path between Start -> Waypoint -> Core.
        5. "Swiss Cheese" method: Randomly delete nodes to create holes and maze-like structure,
           preserving the locked path.
        6. Cleanup disjoint nodes.
        7. Place Tokens (Defense, Utility, Data Caches) based on distance and strategic heuristics.

        Args:
            seed (int): Random seed for generation to ensure reproducibility.
                        If None, the random state is not reset (uses current state).
        
        Returns:
            None: The map is generated in-place within the self.system object.
        """
        # Import here to avoid circular dependency with game.models which imports MapGenerator
        from game.models import Node, BfsNode, Core, Firewall, AntiVirus, RestoNode, Suppressor, \
                                Utility, SelfRepair, KernelRot, PolymorphicShield, SecondaryVector, DataCache

        # Set the seed for reproducible map generation
        if seed is not None:
            random.seed(seed)

        # ---------------------------------------------------------
        # 1. Initialize Grid
        # Create a 2D array of Node objects representing the full grid.
        # ---------------------------------------------------------
        self.system.nodes = [] # Clear existing nodes if any
        for row_index in range(self.system.height):
            row = [None] * self.system.width
            self.system.nodes.append(row)
            for column_index in range(self.system.width):
                # Create a new Node at each coordinate
                self.system.nodes[row_index][column_index] = Node(row_index, column_index)

        # ---------------------------------------------------------
        # 2. Select Start Node
        # Pick a random node on the edges of the map as the entry point.
        # ---------------------------------------------------------
        starting_node = self.system.get_starting_node()
        # The starting node is always visited/exposed initially
        self.system.visit_node(starting_node, force=True)

        # ---------------------------------------------------------
        # 3. Determine Key Locations (Waypoint & Core)
        # ---------------------------------------------------------
        
        # Select a 'waypoint' node at a specific distance (5 jumps) from start.
        # This helps guide the path generation.
        node_at_jumps = self.system.get_nodes_at_jumps(starting_node, 5)
        waypoint = random.choice(node_at_jumps)

        # Improved Core Selection Logic:
        # The Core should be far from the start and ideally central, but hard to reach.
        best_candidates = []
        best_score = -1.0
        
        # Calculate center of the map for scoring
        center_row = (self.system.height - 1) / 2.0
        center_col = (self.system.width - 1) / 2.0
        
        for row in range(self.system.height):
            for col in range(self.system.width):
                node = self.system.nodes[row][col]
                sp = self.system.get_path(starting_node, node)
                
                if sp is None:
                    continue
                    
                dist = len(sp)
                if dist < 8: # Hard constraint: Core must be at least 8 steps away from Start
                    continue
                
                # Score calculation: Prefer greater distance (weight 4.0) and distance from center.
                # The formula favors nodes that are far from start AND far from center (edges/corners).
                dist_from_center_sq = (row - center_row)**2 + (col - center_col)**2
                score = (dist * 4.0) + dist_from_center_sq
                
                if score > best_score:
                    best_score = score
                    best_candidates = [node]
                elif abs(score - best_score) < 0.1:
                    best_candidates.append(node)

        # Fallback: If no ideal candidate found, just pick any node far enough
        if not best_candidates:
             for row in range(self.system.height):
                for col in range(self.system.width):
                     node = self.system.nodes[row][col]
                     sp = self.system.get_path(starting_node, node)
                     if sp and len(sp) >= 8:
                         best_candidates.append(node)
        
        # Place the Core
        if best_candidates:
            core_node = random.choice(best_candidates)
        else:
             # Ultimate fallback: Opposite corner
             core_node = self.system.nodes[self.system.height-1][self.system.width-1] if starting_node.row == 0 else self.system.nodes[0][0]

        self.system.core = Core(self.system)
        core_node.token = self.system.core

        # ---------------------------------------------------------
        # 4. Lock Essential Path
        # Ensure there is at least one valid path from Start -> Waypoint -> Core
        # These nodes will be protected from deletion.
        # ---------------------------------------------------------
        p = self.system.get_path(starting_node, waypoint)
        q = self.system.get_path(waypoint, self.system.core.node)
        locked_nodes = set(p + q)
        locked_nodes.add(starting_node)

        # ---------------------------------------------------------
        # 5. Random Deletion (The "Swiss Cheese" method)
        # Randomly remove nodes to create obstacles and shape the map.
        # ---------------------------------------------------------
        population = []
        for row in range(self.system.height):
            for col in range(self.system.width):
                node = self.system.nodes[row][col]
                if node in locked_nodes:
                    continue
                population.append(node)

        nodes_to_delete = []
        max_holes = 6 # Maximum number of "holes" (deleted clusters) to create
        
        center_row = self.system.height / 2
        center_col = self.system.width / 2

        for i in range(max_holes):
            if not population:
                break
            
            # Candidate selection for deletion:
            # The first 4 holes prefer central nodes (close to center).
            # Subsequent holes prefer edge nodes (far from center).
            candidates_with_scores = []
            for node in population:
                dist_sq = (node.row - center_row)**2 + (node.column - center_col)**2
                score = random.random() * 20.0 
                
                if i < 4:
                    score += (100.0 - dist_sq) # Higher score for closer to center
                else:
                    score += dist_sq # Higher score for further from center
                
                candidates_with_scores.append((score, node))
            
            # Pick from top 25% candidates to maintain some randomness
            candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
            top_count = max(1, len(candidates_with_scores) // 4) 
            top_candidates = candidates_with_scores[:top_count]
            
            seed_node = random.choice(top_candidates)[1]
            
            # Connectivity check: Don't delete if it isolates neighbors too much
            safe_to_delete = True
            neighbors = self.system.get_neighbors(seed_node)
            for n in neighbors:
                if n and self.system.get_num_neighbors(n) <= 2:
                     safe_to_delete = False
                     break
            
            if not safe_to_delete:
                continue

            # Delete a small cluster (1 or 2 nodes)
            cluster_size = random.randint(1, 2)
            current_hole = [seed_node]
            
            if cluster_size > 1:
                valid_neighbors = [n for n in neighbors if n in population and n != seed_node]
                if valid_neighbors:
                    second_node = random.choice(valid_neighbors)
                    # Check safety for second node
                    safe_second = True
                    sec_neighbors = self.system.get_neighbors(second_node)
                    for sn in sec_neighbors:
                        if sn and sn != seed_node and self.system.get_num_neighbors(sn) <= 2:
                            safe_second = False
                            break
                    
                    if safe_second:
                        current_hole.append(second_node)
            
            # Mark for deletion
            for node in current_hole:
                if node in population:
                    population.remove(node)
                    nodes_to_delete.append(node)
        
        # Apply deletion
        for node in nodes_to_delete:
            self.system.nodes[node.row][node.column] = None

        # ---------------------------------------------------------
        # 6. Cleanup
        # Remove any nodes that became unreachable from the start.
        # ---------------------------------------------------------
        self.system.prune_disjoint(starting_node)

        # ---------------------------------------------------------
        # 7. Place Tokens
        # Populate the map with enemies (Firewall, AntiVirus, etc.) and items.
        # ---------------------------------------------------------
        
        # Get available nodes for placement (excluding Start and Core)
        population = list(filter(lambda n: n is not None and n.token is None, [node for row in self.system.nodes for node in row]))
        if starting_node in population:
            population.remove(starting_node)
            
        # Pre-calculate distances from Core and Start for all nodes
        # This is used to place stronger enemies closer to Core or critical paths.
        dist_from_core = {}
        queue = deque([BfsNode(self.system.core.node, None, 0)])
        visited = {self.system.core.node}
        while queue:
            curr = queue.pop()
            dist_from_core[curr.node] = curr.depth
            for n in self.system.get_neighbors(curr.node):
                if n and n not in visited:
                    visited.add(n)
                    queue.insert(0, BfsNode(n, curr, curr.depth + 1))
        
        dist_from_start = {}
        queue = deque([BfsNode(starting_node, None, 0)])
        visited = {starting_node}
        while queue:
            curr = queue.pop()
            dist_from_start[curr.node] = curr.depth
            for n in self.system.get_neighbors(curr.node):
                if n and n not in visited:
                    visited.add(n)
                    queue.insert(0, BfsNode(n, curr, curr.depth + 1))
                    
        total_dist_start_core = dist_from_start.get(self.system.core.node, 20)
        placed_tokens_nodes = [self.system.core.node]

        def pick_best_node(token_cls, candidates):
            """
            Helper function to select the best node for a specific token type based on heuristics.
            
            Args:
                token_cls (class): The class of the token to be placed.
                candidates (list): List of available Node objects.
                
            Returns:
                Node: The chosen node, or None if no suitable candidate found.
            """
            if not candidates:
                return None
            
            scored_candidates = []
            for node in candidates:
                score = random.random() * 10.0
                
                # Constraint: Avoid clumping.
                # Check distance to already placed tokens.
                too_close = False
                for existing in placed_tokens_nodes:
                    row_diff = abs(node.row - existing.row)
                    col_diff = abs(node.column - existing.column)
                    
                    # Heuristic for "too close": manhattan distance <= 2
                    if row_diff + col_diff <= 2:
                         if node in self.system.get_neighbors(existing) or node == existing:
                             too_close = True
                             break
                         for n in self.system.get_neighbors(existing):
                             if node in self.system.get_neighbors(n):
                                 too_close = True
                                 break
                    if too_close: break
                
                if too_close:
                    score -= 1000.0
                
                d_core = dist_from_core.get(node, 99)
                d_start = dist_from_start.get(node, 99)

                # Never place right next to core (too crowded usually)
                if d_core <= 1:
                    score -= 2000.0

                neighbors = self.system.get_num_neighbors(node)
                
                # --- Type-Specific Scoring Heuristics ---
                
                if token_cls == RestoNode:
                    # RestoNodes like to be near friends (Firewall/AV) to heal them.
                    min_dist_to_friend = 99
                    best_friend_score = 0
                    for r in range(self.system.height):
                        for c in range(self.system.width):
                            n = self.system.nodes[r][c]
                            if n and n.token and (isinstance(n.token, Firewall) or isinstance(n.token, AntiVirus)):
                                p = self.system.get_path(node, n)
                                d = len(p) if p else 99
                                if 2 <= d <= 3: # Sweet spot distance
                                    friend_dist_core = dist_from_core.get(n, 99)
                                    prox_score = 50
                                    strategic_bonus = 0
                                    # Bonus if the friend is protecting the core area
                                    if friend_dist_core < total_dist_start_core * 0.3:
                                        strategic_bonus = 30
                                    current_score = prox_score + strategic_bonus
                                    if current_score > best_friend_score:
                                        best_friend_score = current_score
                    score += best_friend_score
                    
                elif token_cls == Suppressor:
                    # Suppressors block paths. Good at choke points or mid-range.
                    dist_ratio = d_start / max(total_dist_start_core, 1)
                    if 0.2 <= dist_ratio <= 0.5:
                        score += 40
                    elif 0.1 <= dist_ratio <= 0.6:
                        score += 10
                    # Prefer nodes with many connections to maximize blocking effect
                    if neighbors >= 3: score += 30
                    if neighbors >= 4: score += 20
                    # Prefer nodes slightly off the main path
                    path_deviation = (d_start + d_core) - total_dist_start_core
                    if path_deviation >= 2: score += 20
                    
                elif token_cls == Firewall:
                    # Firewalls are high HP blockers.
                    if neighbors >= 3: score += 50
                    elif neighbors >= 4: score += 20
                    if neighbors == 2: score -= 20 # Bad at simple corridors
                    # Good on the main path
                    path_deviation = (d_start + d_core) - total_dist_start_core
                    if path_deviation <= 2: score += 15
                    
                elif token_cls == AntiVirus:
                    # AntiVirus attacks the player. Good on the main path to intercept.
                    path_deviation = (d_start + d_core) - total_dist_start_core
                    if path_deviation == 0: score += 50 # Direct path
                    elif path_deviation <= 2: score += 30
                    elif path_deviation <= 4: score += 10
                    if neighbors >= 3: score += 20
                    
                scored_candidates.append((score, node))
            
            # Pick one of the top 3 candidates
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            valid_candidates = [x for x in scored_candidates if x[0] > -500] # Filter out "too_close" ones
            if not valid_candidates:
                 valid_candidates = scored_candidates
                 
            top_n = min(len(valid_candidates), 3)
            return random.choice(valid_candidates[:top_n])[1]

        # List of standard tokens to spawn
        tokens_to_spawn = [
            Firewall, Firewall, Firewall,
            AntiVirus, AntiVirus, AntiVirus,
            Suppressor, Suppressor, Suppressor,
            RestoNode, RestoNode,
        ]
        
        # Randomly add extra difficulty
        extra_prob = 0.35
        prob_decay = 0.17
        possible_extras = [Firewall, AntiVirus, Suppressor, RestoNode]
        
        while random.random() < extra_prob:
            tokens_to_spawn.append(random.choice(possible_extras))
            extra_prob -= prob_decay

        def type_priority(t):
            """Determines placement order. High value = placed later."""
            if t == Firewall: return 0
            if t == AntiVirus: return 1
            if t == Suppressor: return 2
            if t == RestoNode: return 3
            return 4
            
        tokens_to_spawn.sort(key=type_priority)
        
        # 7.1 Place Outposts (Early/Mid game obstacles)
        outpost_candidates = []
        for node in population:
            d_start = dist_from_start.get(node, 999)
            ratio = d_start / max(total_dist_start_core, 1)
            # Zone: 10% to 50% of the way to core
            if 0.10 <= ratio <= 0.50:
                outpost_candidates.append(node)
        
        # Sort by connectivity (high degree first)
        outpost_candidates.sort(key=lambda n: self.system.get_num_neighbors(n), reverse=True)
        
        num_outposts = 2
        for i in range(num_outposts):
            if not outpost_candidates or not tokens_to_spawn:
                break
            
            target = outpost_candidates.pop(0)
            
            # Prioritize Suppressors or Firewalls for outposts
            token_cls = None
            if Suppressor in tokens_to_spawn: token_cls = Suppressor
            elif Firewall in tokens_to_spawn: token_cls = Firewall
            elif AntiVirus in tokens_to_spawn: token_cls = AntiVirus
            else: token_cls = tokens_to_spawn[0]
            
            tokens_to_spawn.remove(token_cls)
            target.token = token_cls(self.system)
            if target in population:
                population.remove(target)
            placed_tokens_nodes.append(target)
            
            # Don't place outposts too close to each other (remove neighbors from candidates)
            outpost_candidates = [
                n for n in outpost_candidates 
                if n not in self.system.get_neighbors(target) and n != target
            ]

        # 7.2 Place Perimeter Guard (Close to Core)
        # Find nodes exactly 2 steps from Core
        perimeter_nodes = [n for n, d in dist_from_core.items() if d == 2 and n in population]
        
        if perimeter_nodes:
            # Pick the one closest to start (the "front door")
            perimeter_nodes.sort(key=lambda n: dist_from_start.get(n, 999))
            guard_node = perimeter_nodes[0]
            
            if tokens_to_spawn:
                token_cls = None
                if Firewall in tokens_to_spawn: token_cls = Firewall
                elif Suppressor in tokens_to_spawn: token_cls = Suppressor
                elif AntiVirus in tokens_to_spawn: token_cls = AntiVirus
                else: token_cls = tokens_to_spawn[0]
                
                tokens_to_spawn.remove(token_cls)
                guard_node.token = token_cls(self.system)
                population.remove(guard_node)
                placed_tokens_nodes.append(guard_node)
        
        # 7.3 Place Remaining Tokens
        for token_cls in tokens_to_spawn:
            if not population:
                break
            
            target = pick_best_node(token_cls, population)
            if target:
                target.token = token_cls(self.system)
                population.remove(target)
                placed_tokens_nodes.append(target)

        # ---------------------------------------------------------
        # 8. Place Utilities (Player buffs)
        # Utilities are placed in slightly out-of-the-way places to reward exploration.
        # ---------------------------------------------------------
        utility_types = [SelfRepair, KernelRot, PolymorphicShield, SecondaryVector]
        num_utilities = random.randint(2, 4)
        placed_utility_nodes = []
        
        for _ in range(num_utilities):
            if not population: break
            
            candidates = []
            for node in population:
                 d_core = dist_from_core.get(node, 99)
                 d_start = dist_from_start.get(node, 99)
                 # Deviation from optimal path: 2 to 6 steps extra
                 deviation = (d_start + d_core) - total_dist_start_core
                 if 2 <= deviation <= 6:
                     candidates.append(node)
            
            if not candidates:
                candidates = population
            
            sample_size = min(len(candidates), 20)
            sample = random.sample(candidates, sample_size)
            
            best_candidate = None
            best_score = -1
            
            for node in sample:
                score = random.random() * 10.0
                
                # Check spacing
                too_close = False
                for existing in placed_tokens_nodes:
                    row_diff = abs(node.row - existing.row)
                    col_diff = abs(node.column - existing.column)
                    
                    if row_diff + col_diff <= 2: 
                         if node in self.system.get_neighbors(existing) or node == existing:
                             too_close = True
                             break
                         for n in self.system.get_neighbors(existing):
                             if node in self.system.get_neighbors(n):
                                 too_close = True
                                 break
                    if too_close: break
                
                if too_close:
                    score -= 1000.0
                
                # Avoid placing too close to start (player shouldn't get freebies immediately)
                d_start = dist_from_start.get(node, 0)
                early_game_penalty = -50 if d_start < 3 else 0
                score += early_game_penalty
                
                if score > best_score:
                    best_score = score
                    best_candidate = node
            
            # Fallback if all scored poorly
            if best_candidate is None and sample:
                 best_score = -float('inf')
                 for node in sample:
                     score = 0
                     too_close = False
                     for existing in placed_tokens_nodes:
                        row_diff = abs(node.row - existing.row)
                        col_diff = abs(node.column - existing.column)
                        if row_diff + col_diff <= 2: 
                             too_close = True
                             break
                     if too_close: score -= 1000
                     
                     if score > best_score:
                         best_score = score
                         best_candidate = node

            if best_candidate:
                utility_class = random.choice(utility_types)
                best_candidate.token = utility_class(self.system)
                if best_candidate in population:
                    population.remove(best_candidate)
                placed_utility_nodes.append(best_candidate)
                placed_tokens_nodes.append(best_candidate)

        # ---------------------------------------------------------
        # 9. Place Data Caches (Loot)
        # Some defenses/utilities drop data when destroyed.
        # ---------------------------------------------------------
        defense_candidates = []
        utility_candidates = []
        
        for row in range(self.system.height):
            for col in range(self.system.width):
                node = self.system.nodes[row][col]
                if node and node.token:
                    if isinstance(node.token, (Firewall, AntiVirus, Suppressor, RestoNode)):
                        defense_candidates.append(node)
                    elif isinstance(node.token, Utility):
                        utility_candidates.append(node)
        
        targets = []
        
        # Convert 2 defenses and 1 utility into "Data Caches" (wrappers)
        if len(defense_candidates) >= 2:
            targets.extend(random.sample(defense_candidates, 2))
        else:
            targets.extend(defense_candidates)
            
        if utility_candidates:
            targets.append(random.choice(utility_candidates))
            
        for node in targets:
            original_token = node.token
            cache = DataCache(self.system, original_token)
            node.token = cache
