package com.fraudguard360.processing.model;

import java.io.Serializable;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Map;

/**
 * Result of fraud detection processing
 */
public class FraudDetectionResult implements Serializable {
    
    private final String transactionId;
    private final String userId;
    private final Double fraudScore;
    private final String riskLevel; // LOW, MEDIUM, HIGH, CRITICAL
    private final Map<String, Double> featureContributions;
    private final String modelVersion;
    private final LocalDateTime detectionTimestamp;
    private final String reason;
    private final BigDecimal amount;
    private final String location;
    
    public FraudDetectionResult(String transactionId, String userId, Double fraudScore,
                               String riskLevel, Map<String, Double> featureContributions,
                               String modelVersion, String reason, BigDecimal amount, String location) {
        this.transactionId = transactionId;
        this.userId = userId;
        this.fraudScore = fraudScore;
        this.riskLevel = riskLevel;
        this.featureContributions = featureContributions;
        this.modelVersion = modelVersion;
        this.reason = reason;
        this.amount = amount;
        this.location = location;
        this.detectionTimestamp = LocalDateTime.now();
    }
    
    // Getters
    public String getTransactionId() { return transactionId; }
    public String getUserId() { return userId; }
    public Double getFraudScore() { return fraudScore; }
    public String getRiskLevel() { return riskLevel; }
    public Map<String, Double> getFeatureContributions() { return featureContributions; }
    public String getModelVersion() { return modelVersion; }
    public LocalDateTime getDetectionTimestamp() { return detectionTimestamp; }
    public String getReason() { return reason; }
    public BigDecimal getAmount() { return amount; }
    public String getLocation() { return location; }
    
    /**
     * Convert to alert format for Kafka publishing
     */
    public FraudAlert toAlert() {
        return new FraudAlert(
            transactionId,
            userId,
            fraudScore,
            riskLevel,
            reason,
            amount,
            location,
            detectionTimestamp
        );
    }
    
    @Override
    public String toString() {
        return "FraudDetectionResult{" +
                "transactionId='" + transactionId + '\'' +
                ", userId='" + userId + '\'' +
                ", fraudScore=" + fraudScore +
                ", riskLevel='" + riskLevel + '\'' +
                ", detectionTimestamp=" + detectionTimestamp +
                '}';
    }
}