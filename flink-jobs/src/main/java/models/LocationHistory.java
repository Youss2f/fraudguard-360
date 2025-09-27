package models;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Location History for tracking user location patterns and detecting anomalies
 */
public class LocationHistory implements Serializable {
    
    private List<TimestampedLocation> locations;
    private static final int MAX_LOCATION_HISTORY = 500; // Limit memory usage
    
    public LocationHistory() {
        this.locations = new ArrayList<>();
    }
    
    /**
     * Add a new location to the history
     */
    public void addLocation(String location, long timestamp) {
        if (location != null && !location.trim().isEmpty()) {
            locations.add(new TimestampedLocation(location, timestamp));
            
            // Keep only recent locations to prevent memory issues
            if (locations.size() > MAX_LOCATION_HISTORY) {
                locations = locations.stream()
                    .sorted((a, b) -> Long.compare(b.timestamp, a.timestamp))
                    .limit(MAX_LOCATION_HISTORY)
                    .collect(Collectors.toList());
            }
        }
    }
    
    /**
     * Get locations within a time window (in milliseconds)
     */
    public List<String> getLocationsInWindow(long currentTimestamp, long windowSizeMs) {
        long cutoffTime = currentTimestamp - windowSizeMs;
        
        return locations.stream()
            .filter(tl -> tl.timestamp >= cutoffTime)
            .map(tl -> tl.location)
            .collect(Collectors.toList());
    }
    
    /**
     * Get unique locations within a time window
     */
    public Set<String> getUniqueLocationsInWindow(long currentTimestamp, long windowSizeMs) {
        return new HashSet<>(getLocationsInWindow(currentTimestamp, windowSizeMs));
    }
    
    /**
     * Get location changes in time window
     */
    public int getLocationChangesInWindow(long currentTimestamp, long windowSizeMs) {
        List<String> windowLocations = getLocationsInWindow(currentTimestamp, windowSizeMs);
        
        if (windowLocations.size() <= 1) {
            return 0;
        }
        
        int changes = 0;
        String previousLocation = windowLocations.get(0);
        
        for (int i = 1; i < windowLocations.size(); i++) {
            String currentLocation = windowLocations.get(i);
            if (!currentLocation.equals(previousLocation)) {
                changes++;
                previousLocation = currentLocation;
            }
        }
        
        return changes;
    }
    
    /**
     * Get most frequent location in time window
     */
    public String getMostFrequentLocationInWindow(long currentTimestamp, long windowSizeMs) {
        List<String> windowLocations = getLocationsInWindow(currentTimestamp, windowSizeMs);
        
        if (windowLocations.isEmpty()) {
            return null;
        }
        
        return windowLocations.stream()
            .collect(Collectors.groupingBy(loc -> loc, Collectors.counting()))
            .entrySet()
            .stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(null);
    }
    
    /**
     * Get location frequency map in time window
     */
    public Map<String, Long> getLocationFrequencyInWindow(long currentTimestamp, long windowSizeMs) {
        List<String> windowLocations = getLocationsInWindow(currentTimestamp, windowSizeMs);
        
        return windowLocations.stream()
            .collect(Collectors.groupingBy(loc -> loc, Collectors.counting()));
    }
    
    /**
     * Check if location is frequently visited
     */
    public boolean isFrequentLocation(String location, long currentTimestamp, long windowSizeMs, double frequencyThreshold) {
        Map<String, Long> frequencies = getLocationFrequencyInWindow(currentTimestamp, windowSizeMs);
        long totalLocations = frequencies.values().stream().mapToLong(Long::longValue).sum();
        
        if (totalLocations == 0) {
            return false;
        }
        
        long locationCount = frequencies.getOrDefault(location, 0L);
        double frequency = (double) locationCount / totalLocations;
        
        return frequency >= frequencyThreshold;
    }
    
    /**
     * Detect rapid location changes (potential account takeover)
     */
    public boolean hasRapidLocationChanges(long currentTimestamp, long windowSizeMs, int maxChanges) {
        return getLocationChangesInWindow(currentTimestamp, windowSizeMs) > maxChanges;
    }
    
    /**
     * Get location pattern for a specific time range
     */
    public LocationPattern getLocationPattern(long startTime, long endTime) {
        List<TimestampedLocation> rangeLocations = locations.stream()
            .filter(tl -> tl.timestamp >= startTime && tl.timestamp <= endTime)
            .sorted((a, b) -> Long.compare(a.timestamp, b.timestamp))
            .collect(Collectors.toList());
        
        LocationPattern pattern = new LocationPattern();
        
        if (rangeLocations.isEmpty()) {
            return pattern;
        }
        
        // Calculate basic statistics
        Set<String> uniqueLocations = rangeLocations.stream()
            .map(tl -> tl.location)
            .collect(Collectors.toSet());
        
        pattern.uniqueLocationCount = uniqueLocations.size();
        pattern.totalLocationRecords = rangeLocations.size();
        
        // Calculate location changes
        int changes = 0;
        String previousLocation = rangeLocations.get(0).location;
        for (int i = 1; i < rangeLocations.size(); i++) {
            String currentLocation = rangeLocations.get(i).location;
            if (!currentLocation.equals(previousLocation)) {
                changes++;
                previousLocation = currentLocation;
            }
        }
        pattern.locationChanges = changes;
        
        // Find most frequent location
        pattern.mostFrequentLocation = rangeLocations.stream()
            .collect(Collectors.groupingBy(tl -> tl.location, Collectors.counting()))
            .entrySet()
            .stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(null);
        
        // Calculate time spent in each location
        Map<String, Long> timeSpentMap = new HashMap<>();
        for (int i = 0; i < rangeLocations.size() - 1; i++) {
            TimestampedLocation current = rangeLocations.get(i);
            TimestampedLocation next = rangeLocations.get(i + 1);
            long timeSpent = next.timestamp - current.timestamp;
            timeSpentMap.put(current.location, 
                timeSpentMap.getOrDefault(current.location, 0L) + timeSpent);
        }
        pattern.timeSpentPerLocation = timeSpentMap;
        
        return pattern;
    }
    
    /**
     * Check for impossible travel patterns
     */
    public List<ImpossibleTravel> detectImpossibleTravel(long currentTimestamp, long windowSizeMs) {
        List<ImpossibleTravel> impossibleTravels = new ArrayList<>();
        
        List<TimestampedLocation> windowLocations = locations.stream()
            .filter(tl -> tl.timestamp >= (currentTimestamp - windowSizeMs))
            .sorted((a, b) -> Long.compare(a.timestamp, b.timestamp))
            .collect(Collectors.toList());
        
        for (int i = 0; i < windowLocations.size() - 1; i++) {
            TimestampedLocation from = windowLocations.get(i);
            TimestampedLocation to = windowLocations.get(i + 1);
            
            long timeDiff = to.timestamp - from.timestamp;
            double timeHours = timeDiff / (1000.0 * 60.0 * 60.0);
            
            // Simplified distance calculation (in a real system, use proper geolocation)
            double distance = calculateSimpleDistance(from.location, to.location);
            
            // Assume maximum travel speed of 1000 km/h (airplane)
            double maxPossibleDistance = timeHours * 1000.0;
            
            if (distance > maxPossibleDistance && timeHours < 24) { // Within 24 hours
                ImpossibleTravel travel = new ImpossibleTravel(
                    from.location, to.location, from.timestamp, to.timestamp, 
                    distance, timeHours
                );
                impossibleTravels.add(travel);
            }
        }
        
        return impossibleTravels;
    }
    
    /**
     * Simple distance calculation (placeholder for real geolocation)
     */
    private double calculateSimpleDistance(String location1, String location2) {
        if (location1.equals(location2)) {
            return 0.0;
        }
        
        // Very simplified: different countries = 1000km, different cities = 100km
        if (location1.length() >= 2 && location2.length() >= 2) {
            String country1 = location1.substring(0, 2);
            String country2 = location2.substring(0, 2);
            
            if (!country1.equals(country2)) {
                return 1000.0; // Different countries
            } else {
                return 100.0; // Same country, different cities
            }
        }
        
        return 50.0; // Default distance
    }
    
    /**
     * Get historical locations for a user (all time)
     */
    public Set<String> getAllHistoricalLocations() {
        return locations.stream()
            .map(tl -> tl.location)
            .collect(Collectors.toSet());
    }
    
    /**
     * Remove old locations beyond a certain timestamp
     */
    public void removeOldLocations(long cutoffTimestamp) {
        locations.removeIf(tl -> tl.timestamp < cutoffTimestamp);
    }
    
    /**
     * Get total number of location records
     */
    public int size() {
        return locations.size();
    }
    
    /**
     * Check if location history is empty
     */
    public boolean isEmpty() {
        return locations.isEmpty();
    }
    
    /**
     * Inner class to store location with timestamp
     */
    private static class TimestampedLocation implements Serializable {
        final String location;
        final long timestamp;
        
        TimestampedLocation(String location, long timestamp) {
            this.location = location;
            this.timestamp = timestamp;
        }
    }
    
    /**
     * Inner class for location patterns
     */
    public static class LocationPattern implements Serializable {
        public int uniqueLocationCount = 0;
        public int totalLocationRecords = 0;
        public int locationChanges = 0;
        public String mostFrequentLocation = null;
        public Map<String, Long> timeSpentPerLocation = new HashMap<>();
        
        @Override
        public String toString() {
            return String.format(
                "LocationPattern{uniqueLocations=%d, totalRecords=%d, changes=%d, " +
                "mostFrequent='%s', locationsWithTime=%d}",
                uniqueLocationCount, totalLocationRecords, locationChanges,
                mostFrequentLocation, timeSpentPerLocation.size()
            );
        }
    }
    
    /**
     * Inner class for impossible travel detection
     */
    public static class ImpossibleTravel implements Serializable {
        public final String fromLocation;
        public final String toLocation;
        public final long fromTimestamp;
        public final long toTimestamp;
        public final double distance;
        public final double timeHours;
        public final double requiredSpeed;
        
        public ImpossibleTravel(String fromLocation, String toLocation, 
                               long fromTimestamp, long toTimestamp, 
                               double distance, double timeHours) {
            this.fromLocation = fromLocation;
            this.toLocation = toLocation;
            this.fromTimestamp = fromTimestamp;
            this.toTimestamp = toTimestamp;
            this.distance = distance;
            this.timeHours = timeHours;
            this.requiredSpeed = timeHours > 0 ? distance / timeHours : Double.MAX_VALUE;
        }
        
        @Override
        public String toString() {
            return String.format(
                "ImpossibleTravel{from='%s' to='%s', distance=%.1fkm, time=%.1fh, speed=%.1fkm/h}",
                fromLocation, toLocation, distance, timeHours, requiredSpeed
            );
        }
    }
}