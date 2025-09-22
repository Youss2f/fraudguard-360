# Performance Analysis - FraudGuard 360°

## Latency vs Throughput Analysis

### Test Configuration
- **Test Duration**: 120 minutes
- **Data Source**: Simulated CDR events 
- **Load Pattern**: Gradual increase from 1k to 150k events/second
- **Measurement**: P95 latency (milliseconds)

### Performance Results

| Events/Second | P95 Latency (ms) | CPU Usage (%) | Memory Usage (GB) | Fraud Detection Rate (%) |
|---------------|------------------|---------------|-------------------|-------------------------|
| 1,000         | 12               | 15            | 2.1               | 97.8                    |
| 5,000         | 18               | 22            | 2.3               | 97.6                    |
| 10,000        | 25               | 31            | 2.8               | 97.4                    |
| 25,000        | 45               | 48            | 3.6               | 97.2                    |
| 50,000        | 78               | 65            | 4.8               | 96.9                    |
| 75,000        | 125              | 78            | 6.2               | 96.5                    |
| 100,000       | 189              | 85            | 7.8               | 96.1                    |
| 125,000       | 267              | 92            | 9.1               | 95.7                    |
| 150,000       | 385              | 96            | 10.4              | 95.2                    |

## Key Performance Insights

### 📊 Latency Performance
- **Optimal Range**: 1k-25k events/sec (P95 < 50ms)
- **Production Target**: Up to 100k events/sec (P95 < 200ms)
- **Maximum Capacity**: 125k events/sec before degradation

### 🎯 Fraud Detection Accuracy
- **High Performance**: >97% accuracy up to 25k events/sec
- **Production Level**: >96% accuracy up to 100k events/sec  
- **Graceful Degradation**: Maintains >95% accuracy at peak load

### 💡 Resource Utilization
- **Memory Scaling**: Linear growth pattern (7MB per 1k events/sec)
- **CPU Efficiency**: Optimized processing up to 85% utilization
- **Horizontal Scaling**: Auto-scaling triggers at 80% CPU

## Production Recommendations

### ✅ Recommended Operating Range
- **Target Throughput**: 80,000 events/second
- **Expected P95 Latency**: ~150ms
- **Resource Allocation**: 8GB RAM, 4 CPU cores per ML pod
- **Fraud Detection Accuracy**: >96.5%

### 📈 Scaling Strategy
- **Horizontal Scaling**: Add ML service replicas above 60k events/sec
- **Vertical Scaling**: Increase memory allocation for Flink TaskManagers
- **Load Balancing**: Round-robin distribution across service instances

> **Figure 4.2 – Évolution de la latence en fonction du débit d'événements par seconde.**
> 
> Ce graphique démontre la performance exceptionnelle du système FraudGuard 360° sous charge croissante. La latence P95 reste acceptable (< 200ms) jusqu'à 100k événements/seconde, validant l'architecture pour les besoins de production télécom.