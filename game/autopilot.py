"""
AutoPilot Module for EVE Hacking Minigame.

This module implements an AI agent that plays the hacking minigame using a combination of
Heuristic Greedy Search and a State Machine.

Algorithm Strategy:
1.  **State Machine**: Determines the high-level goal (e.g., EXPLORE, ATTACK, HEAL).
2.  **Heuristic Scoring**: Assigns weights to potential actions based on:
    -   Distance hints (Hot/Cold logic).
    -   Safety/Topology (Rule of Six, Connectivity).
    -   Resource Status (Current Health vs. Threat Level).
    -   Entity Type (Priority targets like Suppressors/RestoNodes).

Architecture Role:
    - AI Agent: Simulates player input to solve the puzzle.
"""

import math
from typing import List, Dict, Optional, Union, Tuple
from .models import System, Node, Virus, Core, Firewall, AntiVirus, RestoNode, Suppressor, Utility, SelfRepair, KernelRot, PolymorphicShield, SecondaryVector

# --- Configuration & Tuning ---
class Config:
    """Centralized configuration for AI tuning parameters and magic numbers."""
    
    # Weights for heuristic scoring
    WEIGHTS = {
        'DISTANCE_HINT': 10.0,    # Weight for moving towards smaller distance numbers
        'UNKNOWN_NEIGHBORS': 2.0, # Weight for exploring nodes with more unknown neighbors (Info Gain)
        'UTILITY_LOOT': 50.0,     # Weight for picking up items
        'ATTACK_CORE': 100.0,     # Priority for attacking the core if killable
        'ATTACK_THREAT': 20.0,    # Priority for clearing threats blocking paths
        'AVOID_RISK': -10.0,      # Negative weight for risky moves when low health
    }
    
    # Combat Thresholds
    ATTACK_THRESHOLD_HIGH_HP = 10.0  # Score threshold to attack when healthy
    ATTACK_THRESHOLD_LOW_HP = 30.0   # Score threshold to attack when injured (more conservative)
    VIRUS_HEALTH_HIGH = 50           # Health considered "High"
    VIRUS_HEALTH_CRITICAL = 20       # Health considered "Critical"
    
    # Scoring Values
    SCORE_MUST_CLICK = 1000.0        # Distance hint 1 -> Must click
    SCORE_SUICIDE = -1000.0          # Score for actions that kill the virus
    SCORE_RISKY = -50.0              # Penalty for leaving virus at critical health
    
    # Target Priorities (Bonus Points)
    PRIORITY_SUPPRESSOR = 60.0       # Increased priority (was 50) - restores strength
    PRIORITY_RESTO = 50.0            # Increased priority (was 40) - stops healing
    PRIORITY_ANTIVIRUS = 20.0
    PRIORITY_FIREWALL = 10.0
    
    # Utility Logic
    HEAL_THRESHOLD = 85              # Use SelfRepair if health <= this
    NUKE_HP_THRESHOLD = 60           # Use nuke on threats with HP > this
    NUKE_CORE_PRIORITY = True        # Always try to nuke Core

    @classmethod
    def update(cls, params: Dict[str, Union[float, int]]):
        """
        Dynamically updates configuration parameters.
        
        Args:
            params (dict): Dictionary of parameter names and values.
                           Can include keys from WEIGHTS or class attributes.
        """
        for key, value in params.items():
            if key in cls.WEIGHTS:
                cls.WEIGHTS[key] = float(value)
            elif hasattr(cls, key):
                # Update class attribute if it exists
                current_type = type(getattr(cls, key))
                setattr(cls, key, current_type(value))

    @classmethod
    def try_load_optimized_params(cls):
        """Attempts to load optimized parameters from best_autopilot_params.json."""
        import json
        import os
        
        # Potential locations for the params file
        # 1. Current working directory
        # 2. Project root (assuming this file is in game/)
        possible_paths = [
            'best_autopilot_params.json',
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_autopilot_params.json')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        params = json.load(f)
                    cls.update(params)
                    return True
                except Exception:
                    continue
        return False

# Attempt to load params on module import
Config.try_load_optimized_params()



class AutoPilot:
    def __init__(self, system: System):
        self.system = system
        self.virus = system.virus
        self.node_scores = {} 
        self.last_action = "Initialized"

    def step(self) -> bool:
        """
        Executes a single step of the AI.
        
        Returns:
            bool: True if an action was taken, False if no valid moves (Stuck/Game Over).
        """
        if self.virus.is_dead:
            self.last_action = "Virus Dead"
            return False

        # 1. Use Instant Utilities (SelfRepair, PolymorphicShield)
        if self._use_instant_utilities():
            self.last_action = "Used Instant Utility"
            return True

        # 2. Identify all interactable nodes (Frontier)
        frontier = self._get_frontier()
        if not frontier:
            self.last_action = "No Valid Moves"
            return False

        # 3. Categorize Frontier Nodes
        targets = self._categorize_targets(frontier)
        
        # 4. Decision Logic (State Machine)
        
        # 4a. PRIORITY: Win Game
        if self._attempt_win(targets['core']):
            self.last_action = "Attacking Core"
            return True

        # 4b. PRIORITY: Loot
        if self._attempt_loot(targets['utility']):
            self.last_action = "Looting Utility"
            return True

        # 4c. PRIORITY: Clear Threats
        if self._attempt_combat(targets['threat']):
            self.last_action = "Engaging Threat"
            return True
            
        # 4d. DEFAULT: Explore Unknowns
        if self._attempt_explore(targets['unknown']):
            self.last_action = "Exploring Unknown"
            return True
            
        # 4e. Fallback: Desperate Attack
        # If we decided not to attack any threats earlier (due to threshold), but have no other moves,
        # we might as well attack the best option if available to avoid getting stuck.
        if self._attempt_combat(targets['threat'], force=True):
             self.last_action = "Desperate Attack"
             return True

        self.last_action = "Stuck"
        return False

    def _get_frontier(self) -> List[Node]:
        """Returns list of nodes that are exposed and can be visited."""
        frontier = []
        for row in self.system.nodes:
            for node in row:
                if node and self.system.can_visit_node(node):
                     frontier.append(node)
        return frontier

    def _categorize_targets(self, frontier: List[Node]) -> Dict[str, List[Node]]:
        """Categorizes frontier nodes into logical groups."""
        targets = {
            'core': [],
            'threat': [],
            'utility': [],
            'unknown': []
        }
        
        for node in frontier:
            # Anti-Cheating: Only categorize as specific types if the node has been visited (revealed).
            # In EVE Hacking, exposed nodes [ ] are unknown until clicked/visited.
            # Exception: Debug mode allows peeking.
            is_revealed = node.is_visited or self.system.debug_mode

            if is_revealed and node.token:
                if isinstance(node.token, Core):
                    targets['core'].append(node)
                elif isinstance(node.token, (Firewall, AntiVirus, RestoNode, Suppressor)):
                    targets['threat'].append(node)
                elif isinstance(node.token, Utility):
                    targets['utility'].append(node)
                else:
                    # Fallback for unknown tokens
                    targets['threat'].append(node)
            else:
                # Treat unvisited exposed nodes as unknown, forcing the AI to 'explore' (click) them first.
                targets['unknown'].append(node)
        return targets

    def _use_instant_utilities(self) -> bool:
        """Automatically uses SelfRepair and PolymorphicShield."""
        used_action = False
        # Copy list to avoid modification during iteration issues
        for utility in self.virus.utilities[:]:
            try:
                if isinstance(utility, SelfRepair):
                    if self.virus.coherence <= Config.HEAL_THRESHOLD: 
                        self.virus.use_utility(utility)
                        used_action = True
                elif isinstance(utility, PolymorphicShield):
                    self.virus.use_utility(utility)
                    used_action = True
            except Exception:
                # Swallow exceptions to prevent AI crash during utility usage
                continue
                
        return used_action

    def _attempt_win(self, core_nodes: List[Node]) -> bool:
        """Prioritize destroying the core."""
        for node in core_nodes:
            # Try to nuke it first
            if self._apply_offensive_utility(node):
                return True
                
            if self._can_kill(node):
                self.system.visit_node(node)
                return True
        return False

    def _attempt_loot(self, utility_nodes: List[Node]) -> bool:
        """Pick up safe utilities."""
        if utility_nodes:
            # Pick the first one (Greedy).
            # Improvement: Could pick the one that exposes most new nodes.
            best_loot = utility_nodes[0]
            self.system.visit_node(best_loot)
            return True
        return False

    def _attempt_combat(self, threat_nodes: List[Node], force: bool = False) -> bool:
        """Evaluates and executes attacks on threats."""
        best_attack = None
        best_attack_score = -float('inf')

        for node in threat_nodes:
            # Check if we should nuke high value threats
            if not force and self._should_nuke(node):
                if self._apply_offensive_utility(node):
                    return True
                    
            if self._can_kill(node):
                score = self._evaluate_threat_kill(node)
                if score > best_attack_score:
                    best_attack_score = score
                    best_attack = node
        
        # Determine threshold
        if force:
            threshold = -float('inf') # Attack anything if desperate
        else:
            threshold = Config.ATTACK_THRESHOLD_HIGH_HP if self.virus.coherence > Config.VIRUS_HEALTH_HIGH else Config.ATTACK_THRESHOLD_LOW_HP
            
        if best_attack and best_attack_score > threshold:
             self.system.visit_node(best_attack)
             return True
        return False

    def _attempt_explore(self, unknown_nodes: List[Node]) -> bool:
        """Selects the best unknown node to explore."""
        best_explore = None
        best_explore_score = -float('inf')

        for node in unknown_nodes:
            score = self._calculate_heuristic(node)
            if score > best_explore_score:
                best_explore_score = score
                best_explore = node

        if best_explore:
            self.system.visit_node(best_explore)
            return True
        return False

    def _apply_offensive_utility(self, target_node: Node) -> bool:
        """Attempts to use KernelRot or SecondaryVector on a target."""
        if not target_node or not target_node.token:
            return False

        # 1. KernelRot (Instant Kill/Big Damage)
        for utility in self.virus.utilities:
            if isinstance(utility, KernelRot):
                # Use on Core or high health threats
                should_use = (isinstance(target_node.token, Core) or 
                              target_node.token.coherence > Config.NUKE_HP_THRESHOLD or
                              isinstance(target_node.token, (Suppressor, RestoNode))) # Priority targets
                
                if should_use:
                    try:
                        self.virus.use_utility(utility, target_node)
                        return True
                    except Exception:
                        continue
        
        # 2. SecondaryVector (DoT)
        for utility in self.virus.utilities:
            if isinstance(utility, SecondaryVector):
                 # Use on Core or high health threats
                 should_use = (isinstance(target_node.token, Core) or 
                               target_node.token.coherence > 40)
                 
                 if should_use:
                    try:
                        self.virus.use_utility(utility, target_node)
                        return True
                    except Exception:
                        continue
        return False

    def _should_nuke(self, node: Node) -> bool:
        """Heuristic: Should we waste a utility on this node?"""
        if not node.token: 
            return False
            
        # Don't nuke if it's nearly dead
        if node.token.coherence <= 20:
            return False

        if isinstance(node.token, (Suppressor, RestoNode)):
            return True
        if node.token.coherence > Config.NUKE_HP_THRESHOLD:
            return True
        return False

    def _can_kill(self, node: Node) -> bool:
        """Deterministic combat simulation."""
        if not node.token:
            return True
        
        # Robustness check
        if not hasattr(node.token, 'coherence') or not hasattr(node.token, 'strength'):
            return True # Assume passable if unknown
        
        virus_strength = self.virus.strength
        target_hp = node.token.coherence
        target_str = node.token.strength
        virus_hp = self.virus.coherence
        
        # Avoid division by zero
        if virus_strength <= 0:
            return False

        turns = math.ceil(target_hp / virus_strength)
        
        # Damage logic
        shield_layers = self.virus.shield_charges
        unmitigated_hits = max(0, (turns - 1) - shield_layers)
        final_damage = unmitigated_hits * target_str
        
        # Must have > 0 HP after fight
        return virus_hp > final_damage

    def _evaluate_threat_kill(self, node: Node) -> float:
        """
        Scoring function for attacking a threat.
        High score = Good target.
        """
        val = 0.0
        token = node.token
        if not token:
            return 0.0
        
        # Robustness check
        if not hasattr(token, 'coherence') or not hasattr(token, 'strength'):
            return 0.0

        # 1. Priority Bonus
        if isinstance(token, Suppressor):
            val += Config.PRIORITY_SUPPRESSOR
        elif isinstance(token, RestoNode):
            val += Config.PRIORITY_RESTO
        elif isinstance(token, Firewall):
            val += Config.PRIORITY_FIREWALL
        elif isinstance(token, AntiVirus):
            val += Config.PRIORITY_ANTIVIRUS
            
        # 2. Cost Calculation
        virus_strength = self.virus.strength
        if virus_strength <= 0:
            return Config.SCORE_SUICIDE

        turns = math.ceil(token.coherence / virus_strength)
        shield_layers = self.virus.shield_charges
        unmitigated_hits = max(0, (turns - 1) - shield_layers)
        final_damage = unmitigated_hits * token.strength
        
        # 3. Heuristic: Damage is bad
        val -= final_damage * 0.5
        
        # 4. Safety Check
        if final_damage >= self.virus.coherence:
            return Config.SCORE_SUICIDE
        
        # Long-term Tradeoff: 
        # If we are low on health, we are risk-averse.
        # BUT, if the target is a Suppressor, killing it might SAVE us in the long run by boosting attack.
        remaining_hp = self.virus.coherence - final_damage
        
        if remaining_hp < Config.VIRUS_HEALTH_CRITICAL:
            # High Risk
            if isinstance(token, (Suppressor, RestoNode)):
                # Worth the risk? Maybe. Penalty is smaller.
                val -= 20.0
            else:
                # Not worth it.
                val += Config.SCORE_RISKY
        
        # 5. Blocking Factor (Information Gain)
        # How many unknown nodes does this threat block?
        blocked_neighbors = 0
        for n in self.system.get_neighbors(node):
            if n and not n.is_visited and not n.is_exposed:
                blocked_neighbors += 1
        
        val += blocked_neighbors * 5.0
        
        return val

    def _calculate_heuristic(self, node: Node) -> float:
        """Calculates the exploration score for an unknown node."""
        score = 0.0
        
        # 1. Distance Hint Logic
        neighbors = self.system.get_neighbors(node)
        min_hint = 6 
        
        for n in neighbors:
            if n.is_visited and n.token is None:
                dist = self.system.get_nearest_meaningful_node_distance(n)
                if dist < min_hint:
                    min_hint = dist
        
        if min_hint == 1:
            score += Config.SCORE_MUST_CLICK
        elif min_hint < 6:
            score += (6 - min_hint) * Config.WEIGHTS['DISTANCE_HINT']
            
        # 2. Information Gain
        unknown_neighbors = 0
        for n in neighbors:
            if n and not n.is_visited and not n.is_exposed:
                unknown_neighbors += 1
        
        score += unknown_neighbors * Config.WEIGHTS['UNKNOWN_NEIGHBORS']
        
        return score
