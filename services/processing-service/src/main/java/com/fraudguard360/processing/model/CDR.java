package com.fraudguard360.processing.model;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Objects;

/**
 * Call Detail Record (CDR) representing a telecom transaction
 * Immutable data class for Flink processing
 */
public class CDR implements Serializable {
    
    @JsonProperty("transaction_id")
    private final String transactionId;
    
    @JsonProperty("user_id")
    private final String userId;
    
    @JsonProperty("call_type")
    private final String callType; // VOICE, SMS, DATA
    
    @JsonProperty("duration_seconds")
    private final Integer durationSeconds;
    
    @JsonProperty("amount")
    private final BigDecimal amount;
    
    @JsonProperty("currency")
    private final String currency;
    
    @JsonProperty("source_number")
    private final String sourceNumber;
    
    @JsonProperty("destination_number")
    private final String destinationNumber;
    
    @JsonProperty("location")
    private final String location;
    
    @JsonProperty("cell_tower_id")
    private final String cellTowerId;
    
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonProperty("timestamp")
    private final LocalDateTime timestamp;
    
    @JsonProperty("device_imei")
    private final String deviceImei;
    
    @JsonProperty("network_type")
    private final String networkType; // 3G, 4G, 5G
    
    @JsonProperty("roaming")
    private final Boolean roaming;
    
    // Constructor for Jackson deserialization
    public CDR() {
        this(null, null, null, null, null, null, null, null, null, null, null, null, null, null);
    }
    
    public CDR(String transactionId, String userId, String callType, Integer durationSeconds,
               BigDecimal amount, String currency, String sourceNumber, String destinationNumber,
               String location, String cellTowerId, LocalDateTime timestamp, String deviceImei,
               String networkType, Boolean roaming) {
        this.transactionId = transactionId;
        this.userId = userId;
        this.callType = callType;
        this.durationSeconds = durationSeconds;
        this.amount = amount;
        this.currency = currency;
        this.sourceNumber = sourceNumber;
        this.destinationNumber = destinationNumber;
        this.location = location;
        this.cellTowerId = cellTowerId;
        this.timestamp = timestamp;
        this.deviceImei = deviceImei;
        this.networkType = networkType;
        this.roaming = roaming;
    }
    
    // Getters
    public String getTransactionId() { return transactionId; }
    public String getUserId() { return userId; }
    public String getCallType() { return callType; }
    public Integer getDurationSeconds() { return durationSeconds; }
    public BigDecimal getAmount() { return amount; }
    public String getCurrency() { return currency; }
    public String getSourceNumber() { return sourceNumber; }
    public String getDestinationNumber() { return destinationNumber; }
    public String getLocation() { return location; }
    public String getCellTowerId() { return cellTowerId; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public String getDeviceImei() { return deviceImei; }
    public String getNetworkType() { return networkType; }
    public Boolean getRoaming() { return roaming; }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CDR cdr = (CDR) o;
        return Objects.equals(transactionId, cdr.transactionId);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(transactionId);
    }
    
    @Override
    public String toString() {
        return "CDR{" +
                "transactionId='" + transactionId + '\'' +
                ", userId='" + userId + '\'' +
                ", callType='" + callType + '\'' +
                ", amount=" + amount +
                ", timestamp=" + timestamp +
                '}';
    }
}