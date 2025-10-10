package com.fraudguard360;

import java.time.Instant;

/**
 * Fraud alert data model
 */
public class FraudAlert {
    private String transactionId;
    private String userId;
    private String alertType;
    private double riskScore;
    private String reason;
    private String severity;
    private Instant timestamp;
    private Instant processingTime;
    private String processorId;

    // Default constructor
    public FraudAlert() {}

    // Getters and setters
    public String getTransactionId() { return transactionId; }
    public void setTransactionId(String transactionId) { this.transactionId = transactionId; }

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }

    public String getAlertType() { return alertType; }
    public void setAlertType(String alertType) { this.alertType = alertType; }

    public double getRiskScore() { return riskScore; }
    public void setRiskScore(double riskScore) { this.riskScore = riskScore; }

    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }

    public String getSeverity() { return severity; }
    public void setSeverity(String severity) { this.severity = severity; }

    public Instant getTimestamp() { return timestamp; }
    public void setTimestamp(Instant timestamp) { this.timestamp = timestamp; }

    public Instant getProcessingTime() { return processingTime; }
    public void setProcessingTime(Instant processingTime) { this.processingTime = processingTime; }

    public String getProcessorId() { return processorId; }
    public void setProcessorId(String processorId) { this.processorId = processorId; }

    @Override
    public String toString() {
        return "FraudAlert{" +
                "transactionId='" + transactionId + '\'' +
                ", userId='" + userId + '\'' +
                ", alertType='" + alertType + '\'' +
                ", riskScore=" + riskScore +
                ", severity='" + severity + '\'' +
                ", reason='" + reason + '\'' +
                '}';
    }
}