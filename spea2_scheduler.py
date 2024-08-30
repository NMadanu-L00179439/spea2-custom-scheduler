import kubernetes.client
from kubernetes import config, watch
import numpy as np
from prometheus_client import start_http_server, Summary, Gauge
import requests
import time

# Define Prometheus metrics
REQUEST_TIME = Summary('scheduler_request_processing_seconds', 'Time spent processing request')
CPU_USAGE = Gauge('scheduler_node_cpu_usage', 'CPU usage of the nodes', ['node'])
ARCHIVE_SIZE = Gauge('scheduler_archive_size', 'Current size of the archive')

class BasicScheduler:
    def __init__(self, pop_size, archive_size, num_objectives):
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.num_objectives = num_objectives
        self.k = max(1, int(np.sqrt(pop_size + archive_size)))
        print(f"Initialized BasicScheduler with pop_size={pop_size}, archive_size={archive_size}, num_objectives={num_objectives}, k={self.k}", flush=True)

    def fetch_node_metrics(self):
        prometheus_url = 'http://prometheus-server.monitoring.svc.cluster.local:80/api/v1/query'
        query = 'node_cpu_seconds_total'

        try:
            response = requests.get(prometheus_url, params={'query': query})
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching Prometheus metrics: {e}", flush=True)
            return {}

        data = response.json()
        cpu_usage = {}
        for result in data['data']['result']:
            node_name = result['metric']['node']
            mode = result['metric']['mode']
            cpu_value = float(result['value'][1])

            if node_name not in cpu_usage:
                cpu_usage[node_name] = {}
            cpu_usage[node_name][mode] = cpu_value

            # Update Prometheus metric
            CPU_USAGE.labels(node=node_name).set(cpu_value)
            print(f"Fetched CPU usage for node {node_name}, mode {mode}: {cpu_value}", flush=True)
        
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

        # Update Prometheus metric
        ARCHIVE_SIZE.set(len(new_archive))

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

@REQUEST_TIME.time()
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
    node_cpu_usage = BasicScheduler(0, 0, 0).fetch_node_metrics()

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
    # Start Prometheus metrics server
    start_http_server(8000)
    
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
