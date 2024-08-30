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
  - **`spea2_scheduler.py`**: Python script implementing the basic scheduler with SPEA2. Script for integrating Prometheus metrics for monitoring and optimization.
  - **`test_basic_scheduler.py`**: Unit tests for validating scheduler functionality and performance.
  
- **`docker/`**: Docker-related files for building and deploying the scheduler.
  - **`Dockerfile`**: Dockerfile for building the custom scheduler image.

- **`kubernetes/`**: Kubernetes configuration files.
  - **`scheduler-deployment.yaml`**: Deployment configuration for the custom scheduler.

## Setup and Installation

**Cluster Creation and Setup**
```
kind create cluster --name spea2-cluster
python3 -m venv spea2-scheduler-env
source spea2-scheduler-env/bin/activate
```

**Dependency Installation**
```
pip install kubernetes requests pyyaml watchdog numpy scipy
```

**Initial Build and Deployment**
```
touch spea2_scheduler.py
docker build -t spea2-scheduler:latest .
kind load docker-image spea2-scheduler:latest --name spea2-cluster
touch deployment.yaml
kubectl apply -f deployment.yaml
```

**Rebuild and Update**
```
docker build -t spea2-scheduler:latest .
docker push spea2-scheduler:latest
kind load docker-image spea2-scheduler:latest --name spea2-cluster
kubectl apply -f deployment.yaml
kubectl get pods -l app=spea2-scheduler
kubectl logs -l app=spea2-scheduler
```

**Role and RoleBinding Setup**
```
touch clusterrole.yaml
touch clusterrolebinding.yaml
kubectl apply -f clusterrole.yaml
kubectl apply -f clusterrolebinding.yaml
```

**Pod Management**
```
kubectl delete pod -l app=spea2-scheduler
kubectl delete pod -l app=my-app
docker rmi spea2-scheduler:latest
kubectl apply -f pod.yaml
kubectl get pods -l app=my-app
kubectl logs -l app=my-app
```

**Additional Deployments**
```
docker build -t spea2-scheduler:latest .
kind load docker-image spea2-scheduler:latest --name spea2-cluster
kubectl apply -f deployment.yaml
kubectl apply -f clusterrole.yaml
kubectl apply -f clusterrolebinding.yaml
kubectl apply -f pod.yaml
kubectl apply -f pod-1.yaml
kubectl apply -f pod-2.yaml
```

**Other Configuration**
```
touch pod.yaml
kubectl apply -f pod.yaml
kind load docker-image spea2-scheduler:latest --name spea2-cluster --delete
kubectl rollout restart deployment spea2-scheduler
```

**Scheduler Role and Binding**
```
touch scheduler-role.yaml
touch scheduler-rolebinding.yaml
kubectl apply -f scheduler-role.yaml
kubectl apply -f scheduler-rolebinding.yaml
```

**Metrics Configuration**
```
touch clusterrole-metrics.yaml
touch clusterrole-binding-metrics.yaml
kubectl apply -f clusterrole-metrics.yaml
kubectl apply -f clusterrole-binding-metrics.yaml
kubectl port-forward svc/prometheus-server 9090:80 -n monitoring
kubectl get services --all-namespaces
```

### Usage

- **Access the Scheduler**: Once deployed, access the custom scheduler through the Kubernetes service. Refer to the service configuration for details on accessing the scheduler's API.

- **Monitor Metrics**: Prometheus metrics can be integrated to monitor the scheduler's performance and optimize CPU usage during pod scheduling. Check the `metrics.py` script for configuration details.

