package com.fraudguard360.processing.model;

import java.io.Serializable;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * User profile containing historical behavior patterns
 */
public class UserProfile implements Serializable {
    
    private final String userId;
    private final BigDecimal averageTransactionAmount;
    private final Integer transactionCountLast30Days;
    private final List<String> frequentLocations;
    private final String riskCategory; // LOW, MEDIUM, HIGH
    private final LocalDateTime accountCreatedDate;
    private final Map<String, Object> behavioralFeatures;
    
    public UserProfile(String userId, BigDecimal averageTransactionAmount,
                      Integer transactionCountLast30Days, List<String> frequentLocations,
                      String riskCategory, LocalDateTime accountCreatedDate,
                      Map<String, Object> behavioralFeatures) {
        this.userId = userId;
        this.averageTransactionAmount = averageTransactionAmount;
        this.transactionCountLast30Days = transactionCountLast30Days;
        this.frequentLocations = frequentLocations;
        this.riskCategory = riskCategory;
        this.accountCreatedDate = accountCreatedDate;
        this.behavioralFeatures = behavioralFeatures;
    }
    
    // Getters
    public String getUserId() { return userId; }
    public BigDecimal getAverageTransactionAmount() { return averageTransactionAmount; }
    public Integer getTransactionCountLast30Days() { return transactionCountLast30Days; }
    public List<String> getFrequentLocations() { return frequentLocations; }
    public String getRiskCategory() { return riskCategory; }
    public LocalDateTime getAccountCreatedDate() { return accountCreatedDate; }
    public Map<String, Object> getBehavioralFeatures() { return behavioralFeatures; }
}