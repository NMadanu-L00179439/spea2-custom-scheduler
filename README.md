# Kubernetes Scheduler Optimization with SPEA2 Algorithm

## Overview

This project focuses on optimizing Kubernetes scheduling using the Strength Pareto Evolutionary Algorithm 2 (SPEA2). The goal is to enhance scheduling efficiency in serverless architectures by integrating SPEA2 into a custom Kubernetes scheduler. The implementation is designed to address challenges in energy efficiency and resource allocation.

## Research Objectives

1. **Energy Consumption Measurement**: Analyze current methodologies for measuring energy consumption in data centers and their impact on scheduling algorithms.
2. **Kubernetes Scheduling Performance**: Evaluate existing Kubernetes scheduling algorithms in serverless environments with respect to energy efficiency.
3. **Optimization Approaches**: Explore and address limitations in current approaches for optimizing energy efficiency in serverless environments.
4. **Improvement in Energy Measurement**: Propose improvements in energy measurement to inform better Kubernetes scheduling.
5. **Comparison of SPEA2-Based Scheduler**: Compare the performance of the SPEA2-based scheduler against existing algorithms in terms of energy efficiency, resource allocation, and application performance.

## Project Structure

- **`scheduler/`**: Contains the custom Kubernetes scheduler implementation using the SPEA2 algorithm.
  - **`basic_scheduler.py`**: Python script implementing the basic scheduler with SPEA2.
  - **`metrics.py`**: Script for integrating Prometheus metrics for monitoring and optimization.
  - **`test_basic_scheduler.py`**: Unit tests for validating scheduler functionality and performance.
  
- **`docker/`**: Docker-related files for building and deploying the scheduler.
  - **`Dockerfile`**: Dockerfile for building the custom scheduler image.
  - **`docker-compose.yml`**: Docker Compose configuration for local development.

- **`kubernetes/`**: Kubernetes configuration files.
  - **`scheduler-deployment.yaml`**: Deployment configuration for the custom scheduler.
  - **`scheduler-service.yaml`**: Service configuration for exposing the scheduler.

- **`docs/`**: Documentation and research-related files.
  - **`implementation.md`**: Detailed description of the implementation, including the SPEA2 algorithm and its integration with Kubernetes.
  - **`test-strategy.md`**: Document outlining the test strategy, results, and analysis for the scheduler.

## Setup and Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
    ```
2. **Build Docker Image**:

```
cd docker
docker build -t your-scheduler-image:latest .
```

3. **Deploy to Kubernetes**

```
kubectl apply -f kubernetes/scheduler-deployment.yaml
kubectl apply -f kubernetes/scheduler-service.yaml
```

4. **Run Tests**

```
cd scheduler
python -m unittest test_basic_scheduler.py
```

### Usage

- **Access the Scheduler**: Once deployed, access the custom scheduler through the Kubernetes service. Refer to the service configuration for details on accessing the scheduler's API.

- **Monitor Metrics**: Prometheus metrics can be integrated to monitor the scheduler's performance and optimize CPU usage during pod scheduling. Check the `metrics.py` script for configuration details.

