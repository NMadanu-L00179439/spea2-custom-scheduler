import kubernetes.client
from kubernetes import config, watch
import numpy as np

class BasicScheduler:
    def __init__(self, pop_size, archive_size, num_objectives):
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.num_objectives = num_objectives
        self.k = max(1, int(np.sqrt(pop_size + archive_size)))
        print(f"Initialized BasicScheduler with pop_size={pop_size}, archive_size={archive_size}, num_objectives={num_objectives}, k={self.k}", flush=True)

    def fetch_node_metrics(self, api):
        k8s_nodes = api.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "nodes")
        cpu_usage = {}
        for stats in k8s_nodes['items']:
            node_name = stats['metadata']['name']
            cpu_usage[node_name] = int(stats['usage']['cpu'].replace('n', '')) / 1e9  # Convert to cores
            mem_usage = int(stats['usage']['memory'].replace('Ki', '')) / 1024  # Convert to MiB
            print(f"Node: {node_name}, CPU Usage: {cpu_usage[node_name]} cores, Memory Usage: {mem_usage} MiB", flush=True)
        return cpu_usage

    def fitness_assignment(self, population, archive, node_cpu_usage):
        combined = np.vstack((population, archive))
        N = len(combined)
        raw_fitness = np.zeros(N)

        for i in range(N):
            raw_fitness[i] = np.sum(np.all(combined <= combined[i], axis=1)) - 1
        
        distances = np.array([1 / (np.sort(np.linalg.norm(combined - combined[i], axis=1))[self.k] + 1e-6) for i in range(N)])
        fitness = raw_fitness + distances
        
        # Normalize CPU usage to be between 0 and 1
        cpu_values = np.array([node_cpu_usage.get(f'node{i}', 0) for i in range(N)])
        cpu_values = (cpu_values - np.min(cpu_values)) / (np.max(cpu_values) - np.min(cpu_values) + 1e-6)
        
        # Incorporate CPU usage into fitness calculation
        fitness -= cpu_values  # Reduce fitness based on CPU usage; this can be adjusted depending on the desired objective
        
        fitness_population = fitness[:len(population)]
        fitness_archive = fitness[len(population):]
        
        # Print fitness values
        print(f"Fitness Population: {fitness_population}", flush=True)
        print(f"Fitness Archive: {fitness_archive}", flush=True)
        
        return fitness_population, fitness_archive

    def environmental_selection(self, population, archive, fitness_population, fitness_archive):
        combined = np.vstack((population, archive))
        fitness_combined = np.hstack((fitness_population, fitness_archive))
        non_dominated_mask = fitness_combined < np.max(fitness_combined)
        new_archive = combined[non_dominated_mask]

        if len(new_archive) > self.archive_size:
            new_archive = new_archive[:self.archive_size]

        print(f"Environmental selection resulted in new archive of size {len(new_archive)}", flush=True)
        return new_archive

    def mating_selection(self, archive):
        if len(archive) < 2:
            print("Archive is too small for mating selection.", flush=True)
            return np.array([])

        selected = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(len(archive), size=2, replace=False)
            selected.append(archive[i] if np.random.rand() < 0.5 else archive[j])
        print(f"Mating pool selected with size {len(selected)}", flush=True)
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
        print(f"Recombination and mutation resulted in new offspring of size {len(offspring)}", flush=True)
        return offspring

    def run(self, node_cpu_usage):
        population = np.random.rand(self.pop_size, self.num_objectives)
        archive = np.random.rand(self.archive_size, self.num_objectives)
        print(f"Initial population: {population}", flush=True)
        print(f"Initial archive: {archive}", flush=True)

        for generation in range(10):
            fitness_population, fitness_archive = self.fitness_assignment(population, archive, node_cpu_usage)
            print(f"Generation {generation}: Fitness Population - {fitness_population}", flush=True)
            print(f"Generation {generation}: Fitness Archive - {fitness_archive}", flush=True)

            archive = self.environmental_selection(population, archive, fitness_population, fitness_archive)
            print(f"Generation {generation}: Archive - {archive}", flush=True)

            if len(archive) == 0:
                print("Archive is empty. Check the SPEA2 implementation.", flush=True)
                return np.array([])

            mating_pool = self.mating_selection(archive)
            print(f"Generation {generation}: Mating Pool - {mating_pool}", flush=True)

            if len(mating_pool) == 0:
                print("Mating pool is empty. Ending the run.", flush=True)
                break

            population = self.recombination_and_mutation(mating_pool)
            print(f"Generation {generation}: New Population - {population}", flush=True)

        return archive

    def filter_pods(self, pods):
        filtered_pods = [pod for pod in pods if "app" in pod.metadata.labels and pod.metadata.labels["app"] == "my-app"]
        print(f"Filtered pods: {filtered_pods}", flush=True)
        return filtered_pods

    def select_node(self, nodes):
        for node in nodes:
            if "kubernetes.io/hostname" in node.metadata.labels:
                hostname = node.metadata.labels["kubernetes.io/hostname"]
                if hostname.endswith("control-plane"):
                    print(f"Selected node: {node.metadata.name}", flush=True)
                    return node
        print("No suitable node found.", flush=True)
        return None

def schedule_pod(pod):
    try:
        config.load_incluster_config()
        print("Loaded in-cluster config", flush=True)
    except kubernetes.config.ConfigException:
        config.load_kube_config()
        print("Loaded kube config", flush=True)

    v1 = kubernetes.client.CoreV1Api()
    api = kubernetes.client.CustomObjectsApi()

    print(f"Running scheduler for all namespaces", flush=True)

    all_nodes = v1.list_node().items
    node_cpu_usage = BasicScheduler(0, 0, 0).fetch_node_metrics(api)

    num_nodes = len(all_nodes)
    pop_size = num_nodes * 5
    archive_size = max(1, pop_size // 5)
    print(f"Population size: {pop_size}", flush=True)
    print(f"Archive size: {archive_size}", flush=True)

    scheduler = BasicScheduler(pop_size=pop_size, archive_size=archive_size, num_objectives=2)

    archive = scheduler.run(node_cpu_usage)
    print(f"Final archive: {archive}", flush=True)

    selected_node = scheduler.select_node(all_nodes)
    if selected_node:
        pod.spec.node_name = selected_node.metadata.name
        v1.patch_namespaced_pod(pod.metadata.name, pod.metadata.namespace, {'spec': {'nodeName': selected_node.metadata.name}})
        print(f"Pod {pod.metadata.name} scheduled on {selected_node.metadata.name}", flush=True)
    else:
        print(f"No suitable node found for pod {pod.metadata.name}", flush=True)

def main():
    print(f"==========================================", flush=True)
    try:
        config.load_incluster_config()
        print("Loaded in-cluster config", flush=True)
    except kubernetes.config.ConfigException:
        config.load_kube_config()
        print("Loaded kube config", flush=True)

    v1 = kubernetes.client.CoreV1Api()
    w = watch.Watch()

    for event in w.stream(v1.list_pod_for_all_namespaces):
        pod = event['object']
        event_type = event['type']

        if event_type == 'ADDED' and pod.spec.node_name is None:
            print(f"New pod detected: {pod.metadata.name} in namespace {pod.metadata.namespace}", flush=True)
            schedule_pod(pod)
    print(f"==========================================", flush=True)

if __name__ == "__main__":
    main()
