import kubernetes.client
from kubernetes import config
import numpy as np

# SPEA2 algorithm implementation
class SPEA2Scheduler:
    def __init__(self, pop_size, archive_size, num_objectives):
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.num_objectives = num_objectives
        self.k = int(np.sqrt(pop_size + archive_size))  # for density estimation

    def fitness_assignment(self, population, archive):
        """Assign fitness to each individual in the population and archive."""
        combined = np.vstack((population, archive))
        N = len(combined)
        strength = np.zeros(N)
        raw_fitness = np.zeros(N)

        # Calculate strength values
        for i in range(N):
            strength[i] = np.sum(np.all(combined[i] <= combined, axis=1)) - 1  # number of solutions it dominates

        # Calculate raw fitness values
        for i in range(N):
            raw_fitness[i] = np.sum(strength[np.all(combined <= combined[i], axis=1)])

        # Calculate density estimation using the k-th nearest neighbor
        distances = np.zeros(N)
        for i in range(N):
            dist = np.sort(np.linalg.norm(combined - combined[i], axis=1))
            distances[i] = 1 / (dist[self.k] + 2)

        # Final fitness value
        fitness = raw_fitness + distances
        return fitness[:len(population)], fitness[len(population):]

    def environmental_selection(self, population, archive, fitness_population, fitness_archive):
        """Select the next generation archive."""
        # Combine population and archive and their fitness
        combined = np.vstack((population, archive))
        fitness_combined = np.hstack((fitness_population, fitness_archive))

        # Select non-dominated solutions
        non_dominated_mask = fitness_combined < 1
        new_archive = combined[non_dominated_mask]

        # If archive is too large, apply truncation
        if len(new_archive) > self.archive_size:
            # Truncate archive to maintain boundary solutions
            distances = np.linalg.norm(new_archive[:, np.newaxis] - new_archive, axis=2)
            np.fill_diagonal(distances, np.inf)
            while len(new_archive) > self.archive_size:
                to_remove = np.argmin(np.min(distances, axis=0))
                new_archive = np.delete(new_archive, to_remove, axis=0)
                distances = np.delete(distances, to_remove, axis=0)
                distances = np.delete(distances, to_remove, axis=1)

        return new_archive

    def mating_selection(self, archive):
        """Select parents for the next generation using binary tournament selection."""
        if len(archive) < 2:
            print("Archive is too small for mating selection.")
            return np.array([])

        selected = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(len(archive), size=2, replace=False)
            selected.append(archive[i] if np.random.rand() < 0.5 else archive[j])
        return np.array(selected)

    def recombination_and_mutation(self, mating_pool):
        """Apply recombination (crossover) and mutation to generate offspring."""
        offspring = np.copy(mating_pool)
        # Apply crossover
        for i in range(0, len(offspring), 2):
            if i + 1 < len(offspring):
                alpha = np.random.rand()
                offspring[i], offspring[i + 1] = (
                    alpha * offspring[i] + (1 - alpha) * offspring[i + 1],
                    alpha * offspring[i + 1] + (1 - alpha) * offspring[i],
                )
        # Apply mutation
        mutation_rate = 0.1
        mutation_strength = 0.1
        mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
        offspring += mutation_strength * mutation_mask * np.random.randn(*offspring.shape)
        return offspring

    def run(self):
        # Initialize population and archive with random values
        population = np.random.rand(self.pop_size, self.num_objectives)
        archive = np.zeros((self.archive_size, self.num_objectives))

        for generation in range(100):  # Replace 100 with your stopping condition
            # Step 2: Fitness assignment
            fitness_population, fitness_archive = self.fitness_assignment(population, archive)
            print(f"Generation {generation}: Fitness Population - {fitness_population}")
            print(f"Generation {generation}: Fitness Archive - {fitness_archive}")

            # Step 3: Environmental selection
            archive = self.environmental_selection(population, archive, fitness_population, fitness_archive)
            print(f"Generation {generation}: Archive - {archive}")

            # Check if the archive has been populated
            if len(archive) == 0:
                print("Archive is empty. Check the SPEA2 implementation.")
                return np.array([])  # Return an empty array if no valid solutions are found

            # Step 5: Mating selection
            mating_pool = self.mating_selection(archive)
            print(f"Generation {generation}: Mating Pool - {mating_pool}")

            # Handle empty mating pool
            if len(mating_pool) == 0:
                print("Mating pool is empty. Ending the run.")
                break

            # Step 6: Recombination and mutation
            population = self.recombination_and_mutation(mating_pool)

        return archive  # Final Pareto-optimal solutions

    def filter_pods(self, pods):
        filtered_pods = []
        for pod in pods:
            if "app" in pod.metadata.labels and pod.metadata.labels["app"] == "my-app":
                filtered_pods.append(pod)
        return filtered_pods

    def select_node(self, nodes):
        for node in nodes:
            if "kubernetes.io/hostname" in node.metadata.labels:
                hostname = node.metadata.labels["kubernetes.io/hostname"]
                if hostname.endswith("node1"):
                    return node
        return None

# Kubernetes interaction
def schedule_pods():
    try:
        # This will load the in-cluster config when running inside a pod
        config.load_incluster_config()
    except kubernetes.config.ConfigException:
        # Fallback to the local kubeconfig if not running inside a pod
       
        config.load_kube_config()

    v1 = kubernetes.client.CoreV1Api()
    scheduler = SPEA2Scheduler(pop_size=100, archive_size=50, num_objectives=2)

    # Get all unscheduled pods
    all_pods = v1.list_pod_for_all_namespaces(watch=False).items
    unscheduled_pods = [pod for pod in all_pods if pod.spec.node_name is None]
    
    # Filter pods according to custom criteria
    filtered_pods = scheduler.filter_pods(unscheduled_pods)

    # Get all nodes
    all_nodes = v1.list_node().items

    # Run SPEA2 to determine optimal scheduling
    archive = scheduler.run()

    for pod in filtered_pods:
        selected_node = scheduler.select_node(all_nodes)
        if selected_node:
            pod.spec.node_name = selected_node.metadata.name
            # Replace create_namespaced_pod with patch if the pod already exists
            v1.patch_namespaced_pod(pod.metadata.name, pod.metadata.namespace, {'spec': {'nodeName': selected_node.metadata.name}})
            print(f"Pod {pod.metadata.name} scheduled on {selected_node.metadata.name}")
        else:
            print(f"No suitable node found for pod {pod.metadata.name}")

if __name__ == "__main__":
    schedule_pods()
