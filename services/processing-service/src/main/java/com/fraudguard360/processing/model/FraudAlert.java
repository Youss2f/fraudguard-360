package com.fraudguard360.processing.model;

import java.io.Serializable;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * Fraud alert for Kafka publishing
 */
public class FraudAlert implements Serializable {
    
    private final String alertId;
    private final String transactionId;
    private final String userId;
    private final Double fraudScore;
    private final String severity;
    private final String description;
    private final BigDecimal amount;
    private final String location;
    private final LocalDateTime timestamp;
    private final String status;
    
    public FraudAlert(String transactionId, String userId, Double fraudScore,
                     String severity, String description, BigDecimal amount,
                     String location, LocalDateTime timestamp) {
        this.alertId = "ALERT-" + System.currentTimeMillis() + "-" + transactionId;
        this.transactionId = transactionId;
        this.userId = userId;
        this.fraudScore = fraudScore;
        this.severity = severity;
        this.description = description;
        this.amount = amount;
        this.location = location;
        this.timestamp = timestamp;
        this.status = "ACTIVE";
    }
    
    // Getters
    public String getAlertId() { return alertId; }
    public String getTransactionId() { return transactionId; }
    public String getUserId() { return userId; }
    public Double getFraudScore() { return fraudScore; }
    public String getSeverity() { return severity; }
    public String getDescription() { return description; }
    public BigDecimal getAmount() { return amount; }
    public String getLocation() { return location; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public String getStatus() { return status; }
}