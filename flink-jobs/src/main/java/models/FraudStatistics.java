package models;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Fraud Statistics for monitoring and reporting
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class FraudStatistics implements Serializable {
    
    @JsonProperty("window_start")
    private long windowStart;
    
    @JsonProperty("window_end")
    private long windowEnd;
    
    @JsonProperty("total_alerts")
    private long totalAlerts;
    
    @JsonProperty("critical_alerts")
    private long criticalAlerts;
    
    @JsonProperty("high_alerts")
    private long highAlerts;
    
    @JsonProperty("medium_alerts")
    private long mediumAlerts;
    
    @JsonProperty("low_alerts")
    private long lowAlerts;
    
    @JsonProperty("alerts_by_type")
    private Map<String, Long> alertsByType;
    
    @JsonProperty("avg_risk_score")
    private double avgRiskScore;
    
    @JsonProperty("max_risk_score")
    private double maxRiskScore;
    
    @JsonProperty("unique_users_affected")
    private long uniqueUsersAffected;
    
    @JsonProperty("total_potential_loss")
    private double totalPotentialLoss;
    
    @JsonProperty("top_fraud_types")
    private Map<String, Long> topFraudTypes;
    
    @JsonProperty("geographical_distribution")
    private Map<String, Long> geographicalDistribution;
    
    public FraudStatistics() {
        this.alertsByType = new HashMap<>();
        this.topFraudTypes = new HashMap<>();
        this.geographicalDistribution = new HashMap<>();
    }
    
    public FraudStatistics(long windowStart, long windowEnd) {
        this();
        this.windowStart = windowStart;
        this.windowEnd = windowEnd;
    }
    
    // Getters and Setters
    public long getWindowStart() { return windowStart; }
    public void setWindowStart(long windowStart) { this.windowStart = windowStart; }
    
    public long getWindowEnd() { return windowEnd; }
    public void setWindowEnd(long windowEnd) { this.windowEnd = windowEnd; }
    
    public long getTotalAlerts() { return totalAlerts; }
    public void setTotalAlerts(long totalAlerts) { this.totalAlerts = totalAlerts; }
    
    public long getCriticalAlerts() { return criticalAlerts; }
    public void setCriticalAlerts(long criticalAlerts) { this.criticalAlerts = criticalAlerts; }
    
    public long getHighAlerts() { return highAlerts; }
    public void setHighAlerts(long highAlerts) { this.highAlerts = highAlerts; }
    
    public long getMediumAlerts() { return mediumAlerts; }
    public void setMediumAlerts(long mediumAlerts) { this.mediumAlerts = mediumAlerts; }
    
    public long getLowAlerts() { return lowAlerts; }
    public void setLowAlerts(long lowAlerts) { this.lowAlerts = lowAlerts; }
    
    public Map<String, Long> getAlertsByType() { return alertsByType; }
    public void setAlertsByType(Map<String, Long> alertsByType) { this.alertsByType = alertsByType; }
    
    public double getAvgRiskScore() { return avgRiskScore; }
    public void setAvgRiskScore(double avgRiskScore) { this.avgRiskScore = avgRiskScore; }
    
    public double getMaxRiskScore() { return maxRiskScore; }
    public void setMaxRiskScore(double maxRiskScore) { this.maxRiskScore = maxRiskScore; }
    
    public long getUniqueUsersAffected() { return uniqueUsersAffected; }
    public void setUniqueUsersAffected(long uniqueUsersAffected) { this.uniqueUsersAffected = uniqueUsersAffected; }
    
    public double getTotalPotentialLoss() { return totalPotentialLoss; }
    public void setTotalPotentialLoss(double totalPotentialLoss) { this.totalPotentialLoss = totalPotentialLoss; }
    
    public Map<String, Long> getTopFraudTypes() { return topFraudTypes; }
    public void setTopFraudTypes(Map<String, Long> topFraudTypes) { this.topFraudTypes = topFraudTypes; }
    
    public Map<String, Long> getGeographicalDistribution() { return geographicalDistribution; }
    public void setGeographicalDistribution(Map<String, Long> geographicalDistribution) { 
        this.geographicalDistribution = geographicalDistribution; 
    }
    
    /**
     * Add an alert to the statistics
     */
    public void addAlert(FraudAlert alert) {
        totalAlerts++;
        
        // Count by severity
        switch (alert.getSeverity()) {
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
        
        // Count by fraud type
        String fraudType = alert.getFraudType();
        alertsByType.put(fraudType, alertsByType.getOrDefault(fraudType, 0L) + 1);
        topFraudTypes.put(fraudType, topFraudTypes.getOrDefault(fraudType, 0L) + 1);
        
        // Update risk scores
        double riskScore = alert.getRiskScore();
        if (riskScore > maxRiskScore) {
            maxRiskScore = riskScore;
        }
        
        // Update geographical distribution
        String location = alert.getLocation();
        if (location != null) {
            geographicalDistribution.put(location, 
                geographicalDistribution.getOrDefault(location, 0L) + 1);
        }
        
        // Estimate potential loss (simplified)
        totalPotentialLoss += estimatePotentialLoss(alert);
    }
    
    /**
     * Finalize statistics calculations
     */
    public void finalizeStatistics() {
        // Calculate average risk score
        if (totalAlerts > 0) {
            // This would need to be calculated during aggregation
            // For now, we'll estimate based on alert distribution
            double weightedScore = (criticalAlerts * 0.9) + (highAlerts * 0.7) + 
                                 (mediumAlerts * 0.5) + (lowAlerts * 0.3);
            avgRiskScore = weightedScore / totalAlerts;
        }
    }
    
    /**
     * Estimate potential financial loss from fraud alert
     */
    private double estimatePotentialLoss(FraudAlert alert) {
        // Simplified loss estimation based on fraud type and risk score
        double baseLoss = 100.0; // Base loss amount
        
        switch (alert.getFraudType()) {
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
        
        // Scale by risk score and severity
        double multiplier = alert.getRiskScore();
        if ("CRITICAL".equals(alert.getSeverity())) {
            multiplier *= 2.0;
        } else if ("HIGH".equals(alert.getSeverity())) {
            multiplier *= 1.5;
        }
        
        return baseLoss * multiplier;
    }
    
    @Override
    public String toString() {
        return String.format(
            "FraudStatistics{totalAlerts=%d, critical=%d, high=%d, medium=%d, low=%d, " +
            "avgRiskScore=%.2f, maxRiskScore=%.2f, uniqueUsers=%d, potentialLoss=%.2f}",
            totalAlerts, criticalAlerts, highAlerts, mediumAlerts, lowAlerts,
            avgRiskScore, maxRiskScore, uniqueUsersAffected, totalPotentialLoss
        );
    }
}