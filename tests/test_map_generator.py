import unittest
from game.models import System, Node, Core, Firewall, AntiVirus, DataCache
from game.map_generator import MapGenerator

class TestMapGenerator(unittest.TestCase):
    def test_map_generation(self):
        """Test that the map generator populates the system with nodes and tokens."""
        seed = 12345
        system = System(seed=seed)
        
        # Verify grid dimensions
        self.assertEqual(len(system.nodes), system.height)
        self.assertEqual(len(system.nodes[0]), system.width)
        
        # Verify some nodes are created (not all None)
        node_count = 0
        for row in system.nodes:
            for node in row:
                if node is not None:
                    node_count += 1
        self.assertTrue(node_count > 0, "Map generator should create nodes")
        
        # Verify Core placement
        core_found = False
        for row in system.nodes:
            for node in row:
                if node and node.token and isinstance(node.token, Core):
                    core_found = True
                    break
        self.assertTrue(core_found, "Map generator should place a Core")

    def test_consistent_seed(self):
        """Test that the same seed produces the same map."""
        seed = 67890
        system1 = System(seed=seed)
        system2 = System(seed=seed)
        
        # Compare node structure
        for r in range(system1.height):
            for c in range(system1.width):
                node1 = system1.nodes[r][c]
                node2 = system2.nodes[r][c]
                
                if node1 is None:
                    self.assertIsNone(node2)
                else:
                    self.assertIsNotNone(node2)
                    self.assertEqual(node1.row, node2.row)
                    self.assertEqual(node1.column, node2.column)
                    
                    # Compare tokens if present
                    if node1.token:
                        self.assertIsNotNone(node2.token)
                        self.assertEqual(type(node1.token), type(node2.token))
                    else:
                        self.assertIsNone(node2.token)

if __name__ == '__main__':
    unittest.main()
