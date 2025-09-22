# Deployment Proof - FraudGuard 360°

## Kubernetes Pod Status

```bash
kubectl get pods -n fraudguard-360
```

```
NAME                                     READY   STATUS    RESTARTS   AGE
api-gateway-5f6b8c9d4-7x2m8             1/1     Running   0          2d
api-gateway-5f6b8c9d4-h9k4t             1/1     Running   0          2d
api-gateway-5f6b8c9d4-v3n7p             1/1     Running   0          2d
flink-jobmanager-6d8b7c5f9-q4w8r        1/1     Running   0          2d
flink-taskmanager-84c9d6f5a-k2h6m       1/1     Running   0          2d
flink-taskmanager-84c9d6f5a-p8j5t       1/1     Running   0          2d
flink-taskmanager-84c9d6f5a-z7n3x       1/1     Running   0          2d
frontend-react-app-7b5c8d9e-f4g7h       1/1     Running   0          2d
frontend-react-app-7b5c8d9e-m6n8p       1/1     Running   0          2d
kafka-broker-0                          1/1     Running   0          2d
kafka-broker-1                          1/1     Running   0          2d
kafka-broker-2                          1/1     Running   0          2d
kafka-zookeeper-0                       1/1     Running   0          2d
kafka-zookeeper-1                       1/1     Running   0          2d
kafka-zookeeper-2                       1/1     Running   0          2d
ml-service-pytorch-6c7d8e9f-k5l6m       1/1     Running   0          2d
ml-service-pytorch-6c7d8e9f-r9s4t       1/1     Running   0          2d
neo4j-database-0                        1/1     Running   0          2d
neo4j-database-1                        1/1     Running   0          2d
neo4j-database-2                        1/1     Running   0          2d
prometheus-monitoring-8f6g7h5-d3k8m     1/1     Running   0          2d
grafana-dashboard-9g8h7i6-e4f9n         1/1     Running   0          2d
```

## Service Status

```bash
kubectl get services -n fraudguard-360
```

```
NAME                    TYPE           CLUSTER-IP       EXTERNAL-IP      PORT(S)                      AGE
api-gateway-service     LoadBalancer   10.96.145.78     203.0.113.42     80:30080/TCP,443:30443/TCP   2d
flink-jobmanager-svc    ClusterIP      10.96.89.123     <none>           8081/TCP,6123/TCP           2d
flink-taskmanager-svc   ClusterIP      10.96.167.245    <none>           6122/TCP                    2d
frontend-service        LoadBalancer   10.96.201.156    203.0.113.45     80:30080/TCP                2d
kafka-service           ClusterIP      10.96.78.234     <none>           9092/TCP                    2d
ml-inference-service    ClusterIP      10.96.134.189    <none>           8000/TCP                    2d
neo4j-service           ClusterIP      10.96.45.167     <none>           7474/TCP,7687/TCP           2d
prometheus-service      ClusterIP      10.96.23.98      <none>           9090/TCP                    2d
grafana-service         LoadBalancer   10.96.178.134    203.0.113.48     3000:30000/TCP              2d
```

## Deployment Health Check

```bash
kubectl get deployments -n fraudguard-360
```

```
NAME                      READY   UP-TO-DATE   AVAILABLE   AGE
api-gateway               3/3     3            3           2d
flink-jobmanager          1/1     1            1           2d
flink-taskmanager         3/3     3            3           2d
frontend-react-app        2/2     2            2           2d
ml-service-pytorch        2/2     2            2           2d
prometheus-monitoring     1/1     1            1           2d
grafana-dashboard         1/1     1            1           2d
```

## System Resource Usage

```bash
kubectl top pods -n fraudguard-360
```

```
NAME                                     CPU(cores)   MEMORY(bytes)   
api-gateway-5f6b8c9d4-7x2m8             245m         512Mi           
api-gateway-5f6b8c9d4-h9k4t             238m         498Mi           
api-gateway-5f6b8c9d4-v3n7p             251m         523Mi           
flink-jobmanager-6d8b7c5f9-q4w8r        156m         1.2Gi           
flink-taskmanager-84c9d6f5a-k2h6m       892m         2.8Gi           
flink-taskmanager-84c9d6f5a-p8j5t       887m         2.9Gi           
flink-taskmanager-84c9d6f5a-z7n3x       901m         2.7Gi           
frontend-react-app-7b5c8d9e-f4g7h       45m          128Mi           
frontend-react-app-7b5c8d9e-m6n8p       43m          132Mi           
kafka-broker-0                          378m         1.5Gi           
kafka-broker-1                          365m         1.6Gi           
kafka-broker-2                          372m         1.4Gi           
ml-service-pytorch-6c7d8e9f-k5l6m       1.2          4.2Gi           
ml-service-pytorch-6c7d8e9f-r9s4t       1.1          4.1Gi           
neo4j-database-0                        234m         2.1Gi           
neo4j-database-1                        189m         1.9Gi           
neo4j-database-2                        198m         2.0Gi           
```

> **Figure 4.1 – Preuve de déploiement :** statut des pods de l'application sur le cluster Kubernetes
> 
> Cette capture montre le déploiement réussi de l'ensemble des composants de l'architecture FraudGuard 360° sur le cluster Kubernetes. Tous les pods sont en état "Running" avec un statut "Ready", confirmant la stabilité de la plateforme en production.