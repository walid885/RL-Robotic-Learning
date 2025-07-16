"""Unit tests for Robot class."""
import unittest
from unittest.mock import Mock, patch
from src.core.robot import Robot

class TestRobot(unittest.TestCase):
    def setUp(self):
        self.robot = Robot("test_description", [0, 0, 1])
    
    def test_initialization(self):
        self.assertEqual(self.robot.description, "test_description")
        self.assertEqual(self.robot.position, [0, 0, 1])
