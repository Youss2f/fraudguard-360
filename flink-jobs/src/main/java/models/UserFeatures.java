package models;

import java.util.Map;
import java.util.HashMap;

/**
 * Aggregated user features for windowed analysis
 */
public class UserFeatures {
    private String userId;
    private long windowStart;
    private long windowEnd;
    private int callCount;
    private int smsCount;
    private int dataSessionCount;
    private long totalDuration;
    private double totalCost;
    private int uniqueCallees;
    private int internationalCalls;
    private int premiumRateCalls;
    private int nightCalls; // Calls between 10PM - 6AM
    private Map<String, Integer> locationCounts;
    private Map<String, Integer> deviceCounts;
    private double avgCallDuration;
    private double callFrequency; // Calls per hour

    public UserFeatures() {
        this.locationCounts = new HashMap<>();
        this.deviceCounts = new HashMap<>();
    }

    public UserFeatures(String userId, long windowStart, long windowEnd) {
        this();
        this.userId = userId;
        this.windowStart = windowStart;
        this.windowEnd = windowEnd;
    }

    /**
     * Calculate derived features
     */
    public void calculateDerivedFeatures() {
        if (callCount > 0) {
            this.avgCallDuration = (double) totalDuration / callCount;
        }
        
        long windowDurationHours = (windowEnd - windowStart) / (1000 * 60 * 60);
        if (windowDurationHours > 0) {
            this.callFrequency = (double) callCount / windowDurationHours;
        }
    }

    /**
     * Calculate risk score based on features
     */
    public double calculateRiskScore() {
        double risk = 0.0;
        
        // High call frequency
        if (callFrequency > 20) risk += 0.3;
        
        // Many international calls
        if (internationalCalls > 10) risk += 0.2;
        
        // Premium rate calls
        if (premiumRateCalls > 5) risk += 0.4;
        
        // Unusual hours activity
        if (nightCalls > callCount * 0.5) risk += 0.1;
        
        // Multiple devices
        if (deviceCounts.size() > 3) risk += 0.2;
        
        // Multiple locations
        if (locationCounts.size() > 5) risk += 0.1;
        
        return Math.min(risk, 1.0);
    }

    // Getters and Setters
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }

    public long getWindowStart() { return windowStart; }
    public void setWindowStart(long windowStart) { this.windowStart = windowStart; }

    public long getWindowEnd() { return windowEnd; }
    public void setWindowEnd(long windowEnd) { this.windowEnd = windowEnd; }

    public int getCallCount() { return callCount; }
    public void setCallCount(int callCount) { this.callCount = callCount; }

    public int getSmsCount() { return smsCount; }
    public void setSmsCount(int smsCount) { this.smsCount = smsCount; }

    public int getDataSessionCount() { return dataSessionCount; }
    public void setDataSessionCount(int dataSessionCount) { this.dataSessionCount = dataSessionCount; }

    public long getTotalDuration() { return totalDuration; }
    public void setTotalDuration(long totalDuration) { this.totalDuration = totalDuration; }

    public double getTotalCost() { return totalCost; }
    public void setTotalCost(double totalCost) { this.totalCost = totalCost; }

    public int getUniqueCallees() { return uniqueCallees; }
    public void setUniqueCallees(int uniqueCallees) { this.uniqueCallees = uniqueCallees; }

    public int getInternationalCalls() { return internationalCalls; }
    public void setInternationalCalls(int internationalCalls) { this.internationalCalls = internationalCalls; }

    public int getPremiumRateCalls() { return premiumRateCalls; }
    public void setPremiumRateCalls(int premiumRateCalls) { this.premiumRateCalls = premiumRateCalls; }

    public int getNightCalls() { return nightCalls; }
    public void setNightCalls(int nightCalls) { this.nightCalls = nightCalls; }

    public Map<String, Integer> getLocationCounts() { return locationCounts; }
    public void setLocationCounts(Map<String, Integer> locationCounts) { this.locationCounts = locationCounts; }

    public Map<String, Integer> getDeviceCounts() { return deviceCounts; }
    public void setDeviceCounts(Map<String, Integer> deviceCounts) { this.deviceCounts = deviceCounts; }

    public double getAvgCallDuration() { return avgCallDuration; }
    public void setAvgCallDuration(double avgCallDuration) { this.avgCallDuration = avgCallDuration; }

    public double getCallFrequency() { return callFrequency; }
    public void setCallFrequency(double callFrequency) { this.callFrequency = callFrequency; }

    @Override
    public String toString() {
        return "UserFeatures{" +
                "userId='" + userId + '\'' +
                ", callCount=" + callCount +
                ", callFrequency=" + callFrequency +
                ", internationalCalls=" + internationalCalls +
                ", premiumRateCalls=" + premiumRateCalls +
                ", riskScore=" + calculateRiskScore() +
                '}';
    }
}