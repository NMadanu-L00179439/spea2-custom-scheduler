import kubernetes.client
from kubernetes import config
import numpy as np

class SPEA2Scheduler:
    def __init__(self, pop_size, archive_size, num_objectives):
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.num_objectives = num_objectives
        self.k = int(np.sqrt(pop_size + archive_size))
        print(f"Initialized SPEA2Scheduler with pop_size={pop_size}, archive_size={archive_size}, num_objectives={num_objectives}, k={self.k}")

    def fitness_assignment(self, population, archive):
        combined = np.vstack((population, archive))
        N = len(combined)
        strength = np.zeros(N)
        raw_fitness = np.zeros(N)

        for i in range(N):
            strength[i] = np.sum(np.all(combined[i] <= combined, axis=1)) - 1
        raw_fitness = np.array([np.sum(strength[np.all(combined <= combined[i], axis=1)]) for i in range(N)])
        distances = np.array([1 / (np.sort(np.linalg.norm(combined - combined[i], axis=1))[self.k] + 1e-6) for i in range(N)])
        fitness = raw_fitness + distances
        return fitness[:len(population)], fitness[len(population):]

    def environmental_selection(self, population, archive, fitness_population, fitness_archive):
        combined = np.vstack((population, archive))
        fitness_combined = np.hstack((fitness_population, fitness_archive))
        non_dominated_mask = fitness_combined < np.max(fitness_combined)
        new_archive = combined[non_dominated_mask]

        if len(new_archive) > self.archive_size:
            distances = np.linalg.norm(new_archive[:, np.newaxis] - new_archive, axis=2)
            np.fill_diagonal(distances, np.inf)
            while len(new_archive) > self.archive_size:
                to_remove = np.argmin(np.min(distances, axis=0))
                new_archive = np.delete(new_archive, to_remove, axis=0)
                distances = np.delete(distances, to_remove, axis=0)
                distances = np.delete(distances, to_remove, axis=1)

        return new_archive

    def mating_selection(self, archive):
        if len(archive) < 2:
            print("Archive is too small for mating selection.")
            return np.array([])

        selected = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(len(archive), size=2, replace=False)
            selected.append(archive[i] if np.random.rand() < 0.5 else archive[j])
        return np.array(selected)

    def recombination_and_mutation(self, mating_pool):
        offspring = np.copy(mating_pool)
        for i in range(0, len(offspring), 2):
            if i + 1 < len(offspring):
                alpha = np.random.rand()
                offspring[i], offspring[i + 1] = (
                    alpha * offspring[i] + (1 - alpha) * offspring[i + 1],
                    alpha * offspring[i + 1] + (1 - alpha) * offspring[i],
                )
        mutation_rate = 0.1
        mutation_strength = 0.1
        mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
        offspring += mutation_strength * mutation_mask * np.random.randn(*offspring.shape)
        return offspring

    def run(self):
        population = np.random.rand(self.pop_size, self.num_objectives)
        archive = np.zeros((self.archive_size, self.num_objectives))
        print(f"Initial population: {population}")
        print(f"Initial archive: {archive}")

        for generation in range(10):
            fitness_population, fitness_archive = self.fitness_assignment(population, archive)
            print(f"Generation {generation}: Fitness Population - {fitness_population}")
            print(f"Generation {generation}: Fitness Archive - {fitness_archive}")

            archive = self.environmental_selection(population, archive, fitness_population, fitness_archive)
            print(f"Generation {generation}: Archive - {archive}")

            if len(archive) == 0:
                print("Archive is empty. Check the SPEA2 implementation.")
                return np.array([])

            mating_pool = self.mating_selection(archive)
            print(f"Generation {generation}: Mating Pool - {mating_pool}")

            if len(mating_pool) == 0:
                print("Mating pool is empty. Ending the run.")
                break

            population = self.recombination_and_mutation(mating_pool)
            print(f"Generation {generation}: New Population - {population}")

        return archive

    def filter_pods(self, pods):
        filtered_pods = [pod for pod in pods if "app" in pod.metadata.labels and pod.metadata.labels["app"] == "my-app"]
        print(f"Filtered pods: {filtered_pods}")
        return filtered_pods

    def select_node(self, nodes):
        for node in nodes:
            if "kubernetes.io/hostname" in node.metadata.labels:
                hostname = node.metadata.labels["kubernetes.io/hostname"]
                if hostname.endswith("control-plane"):
                    print(f"Selected node: {node.metadata.name}")
                    return node
        print("No suitable node found.")
        return None

def schedule_pods():
    try:
        config.load_incluster_config()
        print("Loaded in-cluster config")
    except kubernetes.config.ConfigException:
        config.load_kube_config()
        print("Loaded kube config")

    v1 = kubernetes.client.CoreV1Api()
    namespace = "default"
    print(f"Current namespace: {namespace}")

    all_nodes = v1.list_node().items
    all_pods = v1.list_namespaced_pod(namespace, watch=False).items
    print(f"All nodes: {all_nodes}")
    print(f"All pods in namespace {namespace}: {all_pods}")

    num_nodes = len(all_nodes)
    num_pods = len(all_pods)
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of pods in namespace {namespace}: {num_pods}")

    pop_size = num_nodes * 5
    archive_size = pop_size // 5
    print(f"Population size: {pop_size}")
    print(f"Archive size: {archive_size}")

    scheduler = SPEA2Scheduler(pop_size=pop_size, archive_size=archive_size, num_objectives=2)

    unscheduled_pods = [pod for pod in all_pods if pod.spec.node_name is None]
    print(f"Unscheduled pods: {unscheduled_pods}")

    filtered_pods = scheduler.filter_pods(unscheduled_pods)

    archive = scheduler.run()
    print(f"Final archive: {archive}")

    for pod in filtered_pods:
        selected_node = scheduler.select_node(all_nodes)
        if selected_node:
            pod.spec.node_name = selected_node.metadata.name
            v1.patch_namespaced_pod(pod.metadata.name, pod.metadata.namespace, {'spec': {'nodeName': selected_node.metadata.name}})
            print(f"Pod {pod.metadata.name} scheduled on {selected_node.metadata.name}")
        else:
            print(f"No suitable node found for pod {pod.metadata.name}")

if __name__ == "__main__":
    schedule_pods()
