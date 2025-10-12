package models;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;
import java.util.Objects;

/**
 * Call Detail Record - the core event in our fraud detection system
 */
public class CDR {
    @JsonProperty("id")
    private String id;
    
    @JsonProperty("caller_id")
    private String callerId;
    
    @JsonProperty("callee_id")
    private String calleeId;
    
    @JsonProperty("call_type")
    private String callType;
    
    @JsonProperty("start_time")
    private String startTime;
    
    @JsonProperty("end_time")
    private String endTime;
    
    @JsonProperty("duration")
    private Integer duration;
    
    @JsonProperty("bytes_transmitted")
    private Long bytesTransmitted;
    
    @JsonProperty("location_caller")
    private String locationCaller;
    
    @JsonProperty("location_callee")
    private String locationCallee;
    
    @JsonProperty("tower_id")
    private String towerId;
    
    @JsonProperty("device_imei")
    private String deviceImei;
    
    @JsonProperty("cost")
    private Double cost;
    
    @JsonProperty("country_code")
    private String countryCode;

    public CDR() {}

    // Getters and Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getCallerId() { return callerId; }
    public void setCallerId(String callerId) { this.callerId = callerId; }

    public String getCalleeId() { return calleeId; }
    public void setCalleeId(String calleeId) { this.calleeId = calleeId; }

    public String getCallType() { return callType; }
    public void setCallType(String callType) { this.callType = callType; }

    public String getStartTime() { return startTime; }
    public void setStartTime(String startTime) { this.startTime = startTime; }

    public String getEndTime() { return endTime; }
    public void setEndTime(String endTime) { this.endTime = endTime; }

    public Integer getDuration() { return duration; }
    public void setDuration(Integer duration) { this.duration = duration; }

    public Long getBytesTransmitted() { return bytesTransmitted; }
    public void setBytesTransmitted(Long bytesTransmitted) { this.bytesTransmitted = bytesTransmitted; }

    public String getLocationCaller() { return locationCaller; }
    public void setLocationCaller(String locationCaller) { this.locationCaller = locationCaller; }

    public String getLocationCallee() { return locationCallee; }
    public void setLocationCallee(String locationCallee) { this.locationCallee = locationCallee; }

    public String getTowerId() { return towerId; }
    public void setTowerId(String towerId) { this.towerId = towerId; }

    public String getDeviceImei() { return deviceImei; }
    public void setDeviceImei(String deviceImei) { this.deviceImei = deviceImei; }

    public Double getCost() { return cost; }
    public void setCost(Double cost) { this.cost = cost; }

    public String getCountryCode() { return countryCode; }
    public void setCountryCode(String countryCode) { this.countryCode = countryCode; }

    /**
     * Get event timestamp for Flink processing
     */
    public long getEventTimestamp() {
        try {
            return Instant.parse(startTime).toEpochMilli();
        } catch (Exception e) {
            return System.currentTimeMillis();
        }
    }
    
    /**
     * Get timestamp for processing
     */
    public long getTimestamp() {
        return getEventTimestamp();
    }
    
    /**
     * Get caller location (alias for locationCaller)
     */
    public String getCallerLocation() {
        return locationCaller;
    }
    
    /**
     * Check if this is a roaming call
     */
    public boolean isRoamingFlag() {
        return callType != null && (callType.equals("ROAMING") || callType.equals("INTERNATIONAL"));
    }
    
    /**
     * Get call ID (alias for id)
     */
    public String getCallId() {
        return id;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CDR cdr = (CDR) o;
        return Objects.equals(id, cdr.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "CDR{" +
                "id='" + id + '\'' +
                ", callerId='" + callerId + '\'' +
                ", calleeId='" + calleeId + '\'' +
                ", callType='" + callType + '\'' +
                ", startTime='" + startTime + '\'' +
                ", duration=" + duration +
                ", cost=" + cost +
                '}';
    }
}
