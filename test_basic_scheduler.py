import unittest
import numpy as np
from spea2_scheduler_test import BasicScheduler

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
        
        # Expected fitness values
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
    
    def test_empty_population(self):
        # Test with empty population and archive
        empty_population = np.empty((0, 0))  # Initialize as an empty 2D array
        empty_archive = np.empty((0, 0))    # Initialize as an empty 2D array
        
        # Perform fitness assignment
        fitness_population, fitness_archive = self.scheduler.fitness_assignment(
            empty_population,
            empty_archive,
            self.mock_node_cpu_usage
        )
        
        # Debugging output
        print(f"Fitness Population size: {fitness_population.size}")
        print(f"Fitness Archive size: {fitness_archive.size}")
        
        # Assertions to check that the results are empty
        self.assertEqual(fitness_population.size, 0, "Fitness population size should be 0 for empty input")
        self.assertEqual(fitness_archive.size, 0, "Fitness archive size should be 0 for empty input")

    def test_edge_case_high_cpu_usage(self):
        # Test with high CPU usage values
        high_cpu_usage = {
            f'node{i}': 100.0 for i in range(8)
        }
        fitness_population, fitness_archive = self.scheduler.fitness_assignment(
            self.mock_population,
            self.mock_archive,
            high_cpu_usage
        )
        self.assertTrue(np.all(fitness_population >= 0))
        self.assertTrue(np.all(fitness_archive >= 0))
    
    def test_performance(self):
        # Test performance with a large population
        large_population = np.random.rand(1000, 2)
        large_archive = np.random.rand(100, 2)
        node_cpu_usage = {f'node{i}': np.random.rand() for i in range(1000)}
        
        fitness_population, fitness_archive = self.scheduler.fitness_assignment(
            large_population,
            large_archive,
            node_cpu_usage
        )
        
        self.assertEqual(fitness_population.shape[0], large_population.shape[0])
        self.assertEqual(fitness_archive.shape[0], large_archive.shape[0])
    
    def test_integration_with_other_functions(self):
        # Example integration test if BasicScheduler has other methods
        # Here you would test the interaction between fitness_assignment and other methods
        # of the BasicScheduler class.
        # This is a placeholder for demonstration.
        
        # For instance, if there is a method `optimize` that uses fitness_assignment internally:
        if hasattr(self.scheduler, 'optimize'):
            result = self.scheduler.optimize(self.mock_population, self.mock_archive, self.mock_node_cpu_usage)
            self.assertIsNotNone(result)
            self.assertTrue(isinstance(result, expected_type))  # Replace expected_type with actual type
        else:
            self.skipTest("The BasicScheduler class does not have an optimize method.")

if __name__ == '__main__':
    unittest.main()
