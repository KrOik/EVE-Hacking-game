import unittest
import time
from game.models import SnowflakeIDGenerator, Node

class TestSnowflakeIDGenerator(unittest.TestCase):
    def test_unique_ids(self):
        """Test that generated IDs are unique."""
        generator = SnowflakeIDGenerator()
        ids = set()
        for _ in range(1000):
            new_id = generator.generate_id()
            self.assertNotIn(new_id, ids)
            ids.add(new_id)

    def test_increasing_ids(self):
        """Test that IDs are strictly increasing."""
        generator = SnowflakeIDGenerator()
        last_id = 0
        for _ in range(100):
            new_id = generator.generate_id()
            self.assertGreater(new_id, last_id)
            last_id = new_id

    def test_structure(self):
        """Test that ID structure seems valid (positive, 64-bit)."""
        generator = SnowflakeIDGenerator()
        uid = generator.generate_id()
        self.assertGreater(uid, 0)
        self.assertLess(uid, 1 << 63)

class TestNode(unittest.TestCase):
    def test_initialization(self):
        """Test basic node initialization."""
        node = Node(1, 2)
        self.assertEqual(node.row, 1)
        self.assertEqual(node.column, 2)
        self.assertFalse(node.is_visited)
        self.assertFalse(node.is_exposed)
        self.assertEqual(node.block_count, 0)
        self.assertIsNone(node.token)

    def test_blocked_property(self):
        """Test is_blocked property logic."""
        node = Node(0, 0)
        self.assertFalse(node.is_blocked)
        
        node.block_count += 1
        self.assertTrue(node.is_blocked)
        
        node.block_count -= 1
        self.assertFalse(node.is_blocked)

    def test_token_assignment(self):
        """Test token setter/getter and bidirectional relationship."""
        # Mock token class
        class MockToken:
            def __init__(self):
                self.node = None
        
        node = Node(0, 0)
        token = MockToken()
        
        node.token = token
        
        self.assertEqual(node.token, token)
        self.assertEqual(token.node, node)
        
        # Test reassignment
        token2 = MockToken()
        node.token = token2
        
        self.assertEqual(node.token, token2)
        self.assertEqual(token2.node, node)
        self.assertIsNone(token.node) # Previous token should be detached

if __name__ == '__main__':
    unittest.main()
