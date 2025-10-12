package com.fraudguard360;

import java.time.Instant;

/**
 * Enriched transaction with risk scores and additional metadata
 */
public class EnrichedTransaction extends Transaction {
    private double amountRisk;
    private double locationRisk;
    private double merchantRisk;
    private double timeRisk;
    private double deviceRisk;
    private Instant processingTime;

    // Default constructor
    public EnrichedTransaction() {
        super();
    }

    // Constructor from base transaction
    public EnrichedTransaction(Transaction transaction) {
        super(transaction.getTransactionId(), transaction.getUserId(), 
              transaction.getMerchantId(), transaction.getAmount(), transaction.getTimestamp());
        this.setMerchantCategory(transaction.getMerchantCategory());
        this.setDeviceFingerprint(transaction.getDeviceFingerprint());
        this.setIpAddress(transaction.getIpAddress());
        this.setLatitude(transaction.getLatitude());
        this.setLongitude(transaction.getLongitude());
    }

    // Getters and setters for risk scores
    public double getAmountRisk() { return amountRisk; }
    public void setAmountRisk(double amountRisk) { this.amountRisk = amountRisk; }

    public double getLocationRisk() { return locationRisk; }
    public void setLocationRisk(double locationRisk) { this.locationRisk = locationRisk; }

    public double getMerchantRisk() { return merchantRisk; }
    public void setMerchantRisk(double merchantRisk) { this.merchantRisk = merchantRisk; }

    public double getTimeRisk() { return timeRisk; }
    public void setTimeRisk(double timeRisk) { this.timeRisk = timeRisk; }

    public double getDeviceRisk() { return deviceRisk; }
    public void setDeviceRisk(double deviceRisk) { this.deviceRisk = deviceRisk; }

    public Instant getProcessingTime() { return processingTime; }
    public void setProcessingTime(Instant processingTime) { this.processingTime = processingTime; }

    /**
     * Calculate overall risk score from individual risk components
     */
    public double getOverallRiskScore() {
        return (amountRisk * 0.3 + locationRisk * 0.2 + merchantRisk * 0.2 + 
                timeRisk * 0.1 + deviceRisk * 0.2);
    }
}

/**
 * Scored transaction with ML fraud probability
 */
class ScoredTransaction extends EnrichedTransaction {
    private double fraudScore;
    private String modelVersion;
    private Instant scoringTime;

    public ScoredTransaction() {
        super();
    }

    public ScoredTransaction(EnrichedTransaction enriched) {
        super(enriched);
        this.setAmountRisk(enriched.getAmountRisk());
        this.setLocationRisk(enriched.getLocationRisk());
        this.setMerchantRisk(enriched.getMerchantRisk());
        this.setTimeRisk(enriched.getTimeRisk());
        this.setDeviceRisk(enriched.getDeviceRisk());
        this.setProcessingTime(enriched.getProcessingTime());
    }

    public double getFraudScore() { return fraudScore; }
    public void setFraudScore(double fraudScore) { this.fraudScore = fraudScore; }

    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }

    public Instant getScoringTime() { return scoringTime; }
    public void setScoringTime(Instant scoringTime) { this.scoringTime = scoringTime; }
}