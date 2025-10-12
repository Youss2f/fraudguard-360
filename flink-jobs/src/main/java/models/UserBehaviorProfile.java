package models;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * User Behavior Profile for tracking normal user patterns
 */
public class UserBehaviorProfile implements Serializable {
    
    private String userId;
    private long firstSeen;
    private long lastUpdated;
    
    // Call patterns
    private Map<Integer, Integer> hourlyCallCounts; // Hour of day -> call count
    private Map<String, Integer> dayOfWeekCallCounts; // Day -> call count
    private Set<String> frequentDestinations;
    private Set<String> frequentLocations;
    
    // Statistical patterns
    private double avgCallDuration;
    private double avgCallCost;
    private double avgCallsPerDay;
    private long totalCalls;
    
    // Location patterns
    private String primaryLocation;
    private Set<String> homeLocations;
    private Set<String> workLocations;
    
    // Behavioral flags
    private boolean frequentInternationalCaller;
    private boolean frequentRoamer;
    private boolean premiumServiceUser;
    
    public UserBehaviorProfile(String userId) {
        this.userId = userId;
        this.firstSeen = System.currentTimeMillis();
        this.lastUpdated = System.currentTimeMillis();
        
        this.hourlyCallCounts = new ConcurrentHashMap<>();
        this.dayOfWeekCallCounts = new ConcurrentHashMap<>();
        this.frequentDestinations = new HashSet<>();
        this.frequentLocations = new HashSet<>();
        this.homeLocations = new HashSet<>();
        this.workLocations = new HashSet<>();
        
        this.totalCalls = 0;
        this.avgCallDuration = 0.0;
        this.avgCallCost = 0.0;
        this.avgCallsPerDay = 0.0;
    }
    
    /**
     * Update profile with new CDR data
     */
    public void updateProfile(CDR cdr, long timestamp) {
        totalCalls++;
        lastUpdated = timestamp;
        
        // Update time-based patterns
        Calendar cal = Calendar.getInstance();
        cal.setTimeInMillis(timestamp);
        int hour = cal.get(Calendar.HOUR_OF_DAY);
        int dayOfWeek = cal.get(Calendar.DAY_OF_WEEK);
        
        hourlyCallCounts.put(hour, hourlyCallCounts.getOrDefault(hour, 0) + 1);
        dayOfWeekCallCounts.put(String.valueOf(dayOfWeek), 
            dayOfWeekCallCounts.getOrDefault(String.valueOf(dayOfWeek), 0) + 1);
        
        // Update location patterns
        String location = cdr.getCallerLocation();
        if (location != null) {
            frequentLocations.add(location);
            
            // Classify locations based on usage patterns
            if (isBusinessHours(hour)) {
                workLocations.add(location);
            } else {
                homeLocations.add(location);
            }
            
            // Update primary location (most frequent)
            if (primaryLocation == null || 
                Collections.frequency(new ArrayList<>(frequentLocations), location) > 
                Collections.frequency(new ArrayList<>(frequentLocations), primaryLocation)) {
                primaryLocation = location;
            }
        }
        
        // Update destination patterns
        frequentDestinations.add(cdr.getCalleeId());
        
        // Update statistical averages
        updateAverages(cdr);
        
        // Update behavioral flags
        updateBehavioralFlags(cdr);
    }
    
    private void updateAverages(CDR cdr) {
        // Update running averages
        double newAvgDuration = ((avgCallDuration * (totalCalls - 1)) + cdr.getDuration()) / totalCalls;
        double newAvgCost = ((avgCallCost * (totalCalls - 1)) + cdr.getCost()) / totalCalls;
        
        avgCallDuration = newAvgDuration;
        avgCallCost = newAvgCost;
        
        // Calculate calls per day
        long daysSinceFirstSeen = Math.max(1, (lastUpdated - firstSeen) / (24 * 60 * 60 * 1000));
        avgCallsPerDay = (double) totalCalls / daysSinceFirstSeen;
    }
    
    private void updateBehavioralFlags(CDR cdr) {
        // Check for international calling pattern
        if ("INTERNATIONAL".equals(cdr.getCallType())) {
            long internationalCalls = frequentDestinations.stream()
                .filter(dest -> dest.startsWith("+")) // Simplified international check
                .count();
            frequentInternationalCaller = (internationalCalls / (double) totalCalls) > 0.1;
        }
        
        // Check for roaming pattern
        if (cdr.isRoamingFlag()) {
            frequentRoamer = true;
        }
        
        // Check for premium service usage
        if (cdr.getCost() > 50.0) { // High cost threshold
            premiumServiceUser = true;
        }
    }
    
    private boolean isBusinessHours(int hour) {
        return hour >= 9 && hour <= 17; // 9 AM to 5 PM
    }
    
    /**
     * Get locations where user frequently makes calls
     */
    public Set<String> getFrequentLocations() {
        return new HashSet<>(frequentLocations);
    }
    
    /**
     * Check if a call pattern is normal for this user
     */
    public boolean isNormalCallPattern(CDR cdr, long timestamp) {
        Calendar cal = Calendar.getInstance();
        cal.setTimeInMillis(timestamp);
        int hour = cal.get(Calendar.HOUR_OF_DAY);
        
        // Check if this hour is typical for the user
        int callsAtThisHour = hourlyCallCounts.getOrDefault(hour, 0);
        int avgCallsAtThisHour = (int) (totalCalls * 0.042); // Rough average (1/24 hours)
        
        return callsAtThisHour >= avgCallsAtThisHour * 0.1; // At least 10% of average
    }
    
    /**
     * Check if a location is normal for this user
     */
    public boolean isNormalLocation(String location) {
        return frequentLocations.contains(location) || 
               homeLocations.contains(location) || 
               workLocations.contains(location);
    }
    
    /**
     * Get anomaly score for a CDR based on user's normal behavior
     */
    public double getAnomalyScore(CDR cdr, long timestamp) {
        double anomalyScore = 0.0;
        
        // Location anomaly
        if (!isNormalLocation(cdr.getCallerLocation())) {
            anomalyScore += 0.3;
        }
        
        // Time anomaly
        if (!isNormalCallPattern(cdr, timestamp)) {
            anomalyScore += 0.2;
        }
        
        // Duration anomaly
        double durationDeviation = Math.abs(cdr.getDuration() - avgCallDuration) / avgCallDuration;
        if (durationDeviation > 2.0) { // More than 2x average
            anomalyScore += 0.2;
        }
        
        // Cost anomaly
        double costDeviation = Math.abs(cdr.getCost() - avgCallCost) / avgCallCost;
        if (costDeviation > 3.0) { // More than 3x average
            anomalyScore += 0.3;
        }
        
        return Math.min(1.0, anomalyScore);
    }
    
    // Getters and setters
    public String getUserId() { return userId; }
    public long getFirstSeen() { return firstSeen; }
    public long getLastUpdated() { return lastUpdated; }
    public Map<Integer, Integer> getHourlyCallCounts() { return hourlyCallCounts; }
    public Map<String, Integer> getDayOfWeekCallCounts() { return dayOfWeekCallCounts; }
    public Set<String> getFrequentDestinations() { return frequentDestinations; }
    public double getAvgCallDuration() { return avgCallDuration; }
    public double getAvgCallCost() { return avgCallCost; }
    public double getAvgCallsPerDay() { return avgCallsPerDay; }
    public long getTotalCalls() { return totalCalls; }
    public String getPrimaryLocation() { return primaryLocation; }
    public Set<String> getHomeLocations() { return homeLocations; }
    public Set<String> getWorkLocations() { return workLocations; }
    public boolean isFrequentInternationalCaller() { return frequentInternationalCaller; }
    public boolean isFrequentRoamer() { return frequentRoamer; }
    public boolean isPremiumServiceUser() { return premiumServiceUser; }
}