package com.fraudguard360.processing.model;

import java.io.Serializable;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Map;

/**
 * Enriched transaction with additional context for fraud detection
 */
public class EnrichedTransaction implements Serializable {
    
    private final CDR originalCdr;
    private final UserProfile userProfile;
    private final Map<String, Object> enrichmentData;
    private final LocalDateTime processedAt;
    
    public EnrichedTransaction(CDR originalCdr, UserProfile userProfile, 
                             Map<String, Object> enrichmentData) {
        this.originalCdr = originalCdr;
        this.userProfile = userProfile;
        this.enrichmentData = enrichmentData;
        this.processedAt = LocalDateTime.now();
    }
    
    // Getters
    public CDR getOriginalCdr() { return originalCdr; }
    public UserProfile getUserProfile() { return userProfile; }
    public Map<String, Object> getEnrichmentData() { return enrichmentData; }
    public LocalDateTime getProcessedAt() { return processedAt; }
    
    // Convenience methods
    public String getTransactionId() { return originalCdr.getTransactionId(); }
    public String getUserId() { return originalCdr.getUserId(); }
    public BigDecimal getAmount() { return originalCdr.getAmount(); }
    public LocalDateTime getTimestamp() { return originalCdr.getTimestamp(); }
    public String getLocation() { return originalCdr.getLocation(); }
    
    @Override
    public String toString() {
        return "EnrichedTransaction{" +
                "transactionId='" + getTransactionId() + '\'' +
                ", userId='" + getUserId() + '\'' +
                ", amount=" + getAmount() +
                ", processedAt=" + processedAt +
                '}';
    }
}