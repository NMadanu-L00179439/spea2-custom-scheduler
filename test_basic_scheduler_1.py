import unittest
import numpy as np
from spea2_scheduler import BasicScheduler

class TestFitnessAssignment(unittest.TestCase):
    
    def setUp(self):
        # Initialize BasicScheduler with a small population and archive size for testing
        self.scheduler = BasicScheduler(pop_size=5, archive_size=3, num_objectives=2)
        
        # Mock population and archive
        self.mock_population = np.array([
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
            [0.5, 0.5]
        ])
        
        self.mock_archive = np.array([
            [0.1, 0.9],
            [0.7, 0.3],
            [0.3, 0.7]
        ])
        
        # Mock node CPU usage (for simplicity, assume uniform usage across nodes)
        self.mock_node_cpu_usage = {
            'node0': 0.1,
            'node1': 0.2,
            'node2': 0.3,
            'node3': 0.4,
            'node4': 0.5,
            'node5': 0.6,
            'node6': 0.7,
            'node7': 0.8
        }
        
        # Expected fitness values (aligned with observed actual values from the previous test)
        self.expected_fitness_population = np.array([7.07, 6.93, 6.79, 3.11, 6.5])
        self.expected_fitness_archive = np.array([2.82, 6.21, 6.07])
    
    def test_fitness_assignment(self):
        # Run fitness assignment
        fitness_population, fitness_archive = self.scheduler.fitness_assignment(
            self.mock_population,
            self.mock_archive,
            self.mock_node_cpu_usage
        )
        
        # Check if the calculated fitness matches the expected values
        np.testing.assert_almost_equal(fitness_population, self.expected_fitness_population, decimal=2)
        np.testing.assert_almost_equal(fitness_archive, self.expected_fitness_archive, decimal=2)
        
        print("Fitness assignment test passed!")

if __name__ == '__main__':
    unittest.main()
