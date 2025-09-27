package aggregators;

import models.FraudAlert;
import models.FraudStatistics;
import org.apache.flink.api.common.functions.AggregateFunction;

import java.util.HashSet;
import java.util.Set;

/**
 * Aggregator for calculating fraud statistics over time windows
 */
public class FraudStatisticsAggregator implements AggregateFunction<FraudAlert, FraudStatisticsAggregator.Accumulator, FraudStatistics> {
    
    @Override
    public Accumulator createAccumulator() {
        return new Accumulator();
    }
    
    @Override
    public Accumulator add(FraudAlert alert, Accumulator accumulator) {
        accumulator.addAlert(alert);
        return accumulator;
    }
    
    @Override
    public FraudStatistics getResult(Accumulator accumulator) {
        return accumulator.toFraudStatistics();
    }
    
    @Override
    public Accumulator merge(Accumulator a, Accumulator b) {
        return a.merge(b);
    }
    
    /**
     * Accumulator for fraud statistics
     */
    public static class Accumulator {
        private long totalAlerts = 0;
        private long criticalAlerts = 0;
        private long highAlerts = 0;
        private long mediumAlerts = 0;
        private long lowAlerts = 0;
        
        private double totalRiskScore = 0.0;
        private double maxRiskScore = 0.0;
        
        private Set<String> uniqueUsers = new HashSet<>();
        private double totalPotentialLoss = 0.0;
        
        private long windowStart = Long.MAX_VALUE;
        private long windowEnd = Long.MIN_VALUE;
        
        public void addAlert(FraudAlert alert) {
            totalAlerts++;
            
            // Update window bounds
            long timestamp = alert.getTimestampAsLong();
            if (timestamp < windowStart) {
                windowStart = timestamp;
            }
            if (timestamp > windowEnd) {
                windowEnd = timestamp;
            }
            
            // Count by severity
            String severity = alert.getSeverity();
            if (severity != null) {
                switch (severity) {
                    case "CRITICAL":
                        criticalAlerts++;
                        break;
                    case "HIGH":
                        highAlerts++;
                        break;
                    case "MEDIUM":
                        mediumAlerts++;
                        break;
                    case "LOW":
                        lowAlerts++;
                        break;
                }
            }
            
            // Update risk scores
            double riskScore = alert.getRiskScore();
            totalRiskScore += riskScore;
            if (riskScore > maxRiskScore) {
                maxRiskScore = riskScore;
            }
            
            // Track unique users
            if (alert.getUserId() != null) {
                uniqueUsers.add(alert.getUserId());
            }
            
            // Estimate potential loss
            totalPotentialLoss += estimatePotentialLoss(alert);
        }
        
        public FraudStatistics toFraudStatistics() {
            FraudStatistics stats = new FraudStatistics();
            
            if (windowStart != Long.MAX_VALUE) {
                stats.setWindowStart(windowStart);
            }
            if (windowEnd != Long.MIN_VALUE) {
                stats.setWindowEnd(windowEnd);
            }
            
            stats.setTotalAlerts(totalAlerts);
            stats.setCriticalAlerts(criticalAlerts);
            stats.setHighAlerts(highAlerts);
            stats.setMediumAlerts(mediumAlerts);
            stats.setLowAlerts(lowAlerts);
            
            stats.setMaxRiskScore(maxRiskScore);
            if (totalAlerts > 0) {
                stats.setAvgRiskScore(totalRiskScore / totalAlerts);
            }
            
            stats.setUniqueUsersAffected(uniqueUsers.size());
            stats.setTotalPotentialLoss(totalPotentialLoss);
            
            return stats;
        }
        
        public Accumulator merge(Accumulator other) {
            Accumulator merged = new Accumulator();
            
            merged.totalAlerts = this.totalAlerts + other.totalAlerts;
            merged.criticalAlerts = this.criticalAlerts + other.criticalAlerts;
            merged.highAlerts = this.highAlerts + other.highAlerts;
            merged.mediumAlerts = this.mediumAlerts + other.mediumAlerts;
            merged.lowAlerts = this.lowAlerts + other.lowAlerts;
            
            merged.totalRiskScore = this.totalRiskScore + other.totalRiskScore;
            merged.maxRiskScore = Math.max(this.maxRiskScore, other.maxRiskScore);
            
            merged.uniqueUsers = new HashSet<>(this.uniqueUsers);
            merged.uniqueUsers.addAll(other.uniqueUsers);
            
            merged.totalPotentialLoss = this.totalPotentialLoss + other.totalPotentialLoss;
            
            merged.windowStart = Math.min(this.windowStart, other.windowStart);
            merged.windowEnd = Math.max(this.windowEnd, other.windowEnd);
            
            return merged;
        }
        
        private double estimatePotentialLoss(FraudAlert alert) {
            double baseLoss = 100.0;
            
            String fraudType = alert.getFraudType();
            if (fraudType != null) {
                switch (fraudType) {
                    case "PREMIUM_RATE_FRAUD":
                        baseLoss = 500.0;
                        break;
                    case "SIM_BOX_FRAUD":
                        baseLoss = 1000.0;
                        break;
                    case "ROAMING_FRAUD":
                        baseLoss = 300.0;
                        break;
                    case "ACCOUNT_TAKEOVER":
                        baseLoss = 800.0;
                        break;
                    default:
                        baseLoss = 200.0;
                }
            }
            
            double multiplier = alert.getRiskScore();
            String severity = alert.getSeverity();
            if ("CRITICAL".equals(severity)) {
                multiplier *= 2.0;
            } else if ("HIGH".equals(severity)) {
                multiplier *= 1.5;
            }
            
            return baseLoss * multiplier;
        }
    }
}