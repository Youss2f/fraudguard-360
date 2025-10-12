package models;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Call History for tracking user's call patterns in sliding windows
 */
public class CallHistory implements Serializable {
    
    private List<TimestampedCDR> calls;
    private static final int MAX_HISTORY_SIZE = 1000; // Limit memory usage
    
    public CallHistory() {
        this.calls = new ArrayList<>();
    }
    
    /**
     * Add a new call to the history
     */
    public void addCall(CDR cdr, long timestamp) {
        calls.add(new TimestampedCDR(cdr, timestamp));
        
        // Keep only recent calls to prevent memory issues
        if (calls.size() > MAX_HISTORY_SIZE) {
            calls = calls.stream()
                .sorted((a, b) -> Long.compare(b.timestamp, a.timestamp))
                .limit(MAX_HISTORY_SIZE)
                .collect(Collectors.toList());
        }
    }
    
    /**
     * Get calls within a time window (in milliseconds)
     */
    public List<CDR> getCallsInWindow(long currentTimestamp, long windowSizeMs) {
        long cutoffTime = currentTimestamp - windowSizeMs;
        
        return calls.stream()
            .filter(tc -> tc.timestamp >= cutoffTime)
            .map(tc -> tc.cdr)
            .collect(Collectors.toList());
    }
    
    /**
     * Get calls within a specific time range
     */
    public List<CDR> getCallsInRange(long startTime, long endTime) {
        return calls.stream()
            .filter(tc -> tc.timestamp >= startTime && tc.timestamp <= endTime)
            .map(tc -> tc.cdr)
            .collect(Collectors.toList());
    }
    
    /**
     * Get call count in time window
     */
    public int getCallCountInWindow(long currentTimestamp, long windowSizeMs) {
        long cutoffTime = currentTimestamp - windowSizeMs;
        
        return (int) calls.stream()
            .filter(tc -> tc.timestamp >= cutoffTime)
            .count();
    }
    
    /**
     * Get calls by destination in time window
     */
    public Map<String, List<CDR>> getCallsByDestinationInWindow(long currentTimestamp, long windowSizeMs) {
        return getCallsInWindow(currentTimestamp, windowSizeMs)
            .stream()
            .collect(Collectors.groupingBy(CDR::getCalleeId));
    }
    
    /**
     * Get calls by type in time window
     */
    public Map<String, List<CDR>> getCallsByTypeInWindow(long currentTimestamp, long windowSizeMs) {
        return getCallsInWindow(currentTimestamp, windowSizeMs)
            .stream()
            .collect(Collectors.groupingBy(CDR::getCallType));
    }
    
    /**
     * Get total cost in time window
     */
    public double getTotalCostInWindow(long currentTimestamp, long windowSizeMs) {
        return getCallsInWindow(currentTimestamp, windowSizeMs)
            .stream()
            .mapToDouble(CDR::getCost)
            .sum();
    }
    
    /**
     * Get average call duration in time window
     */
    public double getAvgDurationInWindow(long currentTimestamp, long windowSizeMs) {
        List<CDR> windowCalls = getCallsInWindow(currentTimestamp, windowSizeMs);
        if (windowCalls.isEmpty()) {
            return 0.0;
        }
        
        return windowCalls.stream()
            .mapToDouble(CDR::getDuration)
            .average()
            .orElse(0.0);
    }
    
    /**
     * Get unique destinations count in time window
     */
    public int getUniqueDestinationsInWindow(long currentTimestamp, long windowSizeMs) {
        return (int) getCallsInWindow(currentTimestamp, windowSizeMs)
            .stream()
            .map(CDR::getCalleeId)
            .distinct()
            .count();
    }
    
    /**
     * Get roaming calls in time window
     */
    public List<CDR> getRoamingCallsInWindow(long currentTimestamp, long windowSizeMs) {
        return getCallsInWindow(currentTimestamp, windowSizeMs)
            .stream()
            .filter(cdr -> cdr.getCallType().equals("ROAMING") || 
                          cdr.getCallType().equals("INTERNATIONAL"))
            .collect(Collectors.toList());
    }
    
    /**
     * Get premium calls (high cost) in time window
     */
    public List<CDR> getPremiumCallsInWindow(long currentTimestamp, long windowSizeMs, double costThreshold) {
        return getCallsInWindow(currentTimestamp, windowSizeMs)
            .stream()
            .filter(cdr -> cdr.getCost() > costThreshold)
            .collect(Collectors.toList());
    }
    
    /**
     * Get short duration calls in time window
     */
    public List<CDR> getShortCallsInWindow(long currentTimestamp, long windowSizeMs, double durationThreshold) {
        return getCallsInWindow(currentTimestamp, windowSizeMs)
            .stream()
            .filter(cdr -> cdr.getDuration() < durationThreshold)
            .collect(Collectors.toList());
    }
    
    /**
     * Check for call bursts (many calls in short time)
     */
    public boolean hasBurstPattern(long currentTimestamp, long burstWindowMs, int burstThreshold) {
        int callsInBurst = getCallCountInWindow(currentTimestamp, burstWindowMs);
        return callsInBurst > burstThreshold;
    }
    
    /**
     * Get call pattern statistics
     */
    public CallPatternStats getPatternStats(long currentTimestamp, long windowSizeMs) {
        List<CDR> windowCalls = getCallsInWindow(currentTimestamp, windowSizeMs);
        
        if (windowCalls.isEmpty()) {
            return new CallPatternStats();
        }
        
        CallPatternStats stats = new CallPatternStats();
        stats.totalCalls = windowCalls.size();
        stats.uniqueDestinations = getUniqueDestinationsInWindow(currentTimestamp, windowSizeMs);
        stats.totalCost = getTotalCostInWindow(currentTimestamp, windowSizeMs);
        stats.avgDuration = getAvgDurationInWindow(currentTimestamp, windowSizeMs);
        stats.internationalCalls = (int) windowCalls.stream()
            .filter(cdr -> cdr.getCallType().equals("INTERNATIONAL"))
            .count();
        stats.shortCalls = getShortCallsInWindow(currentTimestamp, windowSizeMs, 30).size();
        stats.premiumCalls = getPremiumCallsInWindow(currentTimestamp, windowSizeMs, 50.0).size();
        
        return stats;
    }
    
    /**
     * Remove old calls beyond a certain timestamp
     */
    public void removeOldCalls(long cutoffTimestamp) {
        calls.removeIf(tc -> tc.timestamp < cutoffTimestamp);
    }
    
    /**
     * Get total number of calls in history
     */
    public int size() {
        return calls.size();
    }
    
    /**
     * Check if history is empty
     */
    public boolean isEmpty() {
        return calls.isEmpty();
    }
    
    /**
     * Inner class to store CDR with timestamp
     */
    private static class TimestampedCDR implements Serializable {
        final CDR cdr;
        final long timestamp;
        
        TimestampedCDR(CDR cdr, long timestamp) {
            this.cdr = cdr;
            this.timestamp = timestamp;
        }
    }
    
    /**
     * Inner class for call pattern statistics
     */
    public static class CallPatternStats implements Serializable {
        public int totalCalls = 0;
        public int uniqueDestinations = 0;
        public double totalCost = 0.0;
        public double avgDuration = 0.0;
        public int internationalCalls = 0;
        public int shortCalls = 0;
        public int premiumCalls = 0;
        
        @Override
        public String toString() {
            return String.format(
                "CallPatternStats{totalCalls=%d, uniqueDestinations=%d, totalCost=%.2f, " +
                "avgDuration=%.2f, internationalCalls=%d, shortCalls=%d, premiumCalls=%d}",
                totalCalls, uniqueDestinations, totalCost, avgDuration, 
                internationalCalls, shortCalls, premiumCalls
            );
        }
    }
}