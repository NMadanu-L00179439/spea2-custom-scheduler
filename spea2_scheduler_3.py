import kubernetes.client
from kubernetes import config
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicScheduler:
    def __init__(self, pop_size, archive_size, num_objectives):
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.num_objectives = num_objectives
        self.k = max(1, int(np.sqrt(pop_size + archive_size)))
        logger.info(f"Initialized BasicScheduler with pop_size={pop_size}, archive_size={archive_size}, num_objectives={num_objectives}, k={self.k}")

    def fitness_assignment(self, population, archive):
        combined = np.vstack((population, archive))
        N = len(combined)
        raw_fitness = np.zeros(N)

        for i in range(N):
            raw_fitness[i] = np.sum(np.all(combined <= combined[i], axis=1)) - 1
        distances = np.array([1 / (np.sort(np.linalg.norm(combined - combined[i], axis=1))[self.k] + 1e-6) for i in range(N)])
        fitness = raw_fitness + distances
        return fitness[:len(population)], fitness[len(population):]

    def environmental_selection(self, population, archive, fitness_population, fitness_archive):
        combined = np.vstack((population, archive))
        fitness_combined = np.hstack((fitness_population, fitness_archive))
        non_dominated_mask = fitness_combined < np.max(fitness_combined)
        new_archive = combined[non_dominated_mask]

        if len(new_archive) > self.archive_size:
            new_archive = new_archive[:self.archive_size]

        return new_archive

    def mating_selection(self, archive):
        if len(archive) < 2:
            logger.warning("Archive is too small for mating selection.")
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
        archive = np.random.rand(self.archive_size, self.num_objectives)
        logger.info(f"Initial population: {population}")
        logger.info(f"Initial archive: {archive}")

        for generation in range(10):
            fitness_population, fitness_archive = self.fitness_assignment(population, archive)
            logger.info(f"Generation {generation}: Fitness Population - {fitness_population}")
            logger.info(f"Generation {generation}: Fitness Archive - {fitness_archive}")

            archive = self.environmental_selection(population, archive, fitness_population, fitness_archive)
            logger.info(f"Generation {generation}: Archive - {archive}")

            if len(archive) == 0:
                logger.warning("Archive is empty. Check the SPEA2 implementation.")
                return np.array([])

            mating_pool = self.mating_selection(archive)
            logger.info(f"Generation {generation}: Mating Pool - {mating_pool}")

            if len(mating_pool) == 0:
                logger.warning("Mating pool is empty. Ending the run.")
                break

            population = self.recombination_and_mutation(mating_pool)
            logger.info(f"Generation {generation}: New Population - {population}")

        return archive

    def filter_pods(self, pods):
        filtered_pods = [pod for pod in pods if "app" in pod.metadata.labels and pod.metadata.labels["app"] == "my-app"]
        logger.info(f"Filtered pods: {filtered_pods}")
        return filtered_pods

    def select_node(self, nodes):
        for node in nodes:
            if "kubernetes.io/hostname" in node.metadata.labels:
                hostname = node.metadata.labels["kubernetes.io/hostname"]
                if hostname.endswith("control-plane"):
                    logger.info(f"Selected node: {node.metadata.name}")
                    return node
        logger.warning("No suitable node found.")
        return None

def schedule_pods():
    try:
        config.load_incluster_config()
        logger.info("Loaded in-cluster config")
    except kubernetes.config.ConfigException:
        config.load_kube_config()
        logger.info("Loaded kube config")

    v1 = kubernetes.client.CoreV1Api()
    namespace = "default"
    logger.info(f"Current namespace: {namespace}")

    all_nodes = v1.list_node().items
    all_pods = v1.list_namespaced_pod(namespace, watch=False).items
    logger.info(f"All nodes: {all_nodes}")
    logger.info(f"All pods in namespace {namespace}: {all_pods}")

    num_nodes = len(all_nodes)
    num_pods = len(all_pods)
    logger.info(f"Number of nodes: {num_nodes}")
    logger.info(f"Number of pods in namespace {namespace}: {num_pods}")

    pop_size = num_nodes * 5
    archive_size = max(1, pop_size // 5)
    logger.info(f"Population size: {pop_size}")
    logger.info(f"Archive size: {archive_size}")

    scheduler = BasicScheduler(pop_size=pop_size, archive_size=archive_size, num_objectives=2)

    while True:
        all_pods = v1.list_namespaced_pod(namespace, watch=False).items
        unscheduled_pods = [pod for pod in all_pods if pod.spec.node_name is None]
        logger.info(f"Unscheduled pods: {unscheduled_pods}")

        filtered_pods = scheduler.filter_pods(unscheduled_pods)

        archive = scheduler.run()
        logger.info(f"Final archive: {archive}")

        for pod in filtered_pods:
            selected_node = scheduler.select_node(all_nodes)
            if selected_node:
                pod.spec.node_name = selected_node.metadata.name
                v1.patch_namespaced_pod(pod.metadata.name, pod.metadata.namespace, {'spec': {'nodeName': selected_node.metadata.name}})
                logger.info(f"Pod {pod.metadata.name} scheduled on {selected_node.metadata.name}")
            else:
                logger.warning(f"No suitable node found for pod {pod.metadata.name}")

        time.sleep(60)  # Wait for a minute before the next scheduling run

if __name__ == "__main__":
    schedule_pods()
