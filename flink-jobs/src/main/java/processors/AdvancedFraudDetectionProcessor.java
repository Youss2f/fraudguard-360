package processors;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import models.CDR;
import models.FraudAlert;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Advanced Fraud Detection Processor for real-time CDR analysis
 * Implements multiple fraud detection algorithms and risk scoring
 */
public class AdvancedFraudDetectionProcessor extends KeyedProcessFunction<String, CDR, FraudAlert> {
    
    private static final Logger logger = LoggerFactory.getLogger(AdvancedFraudDetectionProcessor.class);
    
    // Fraud detection thresholds
    private static final double VELOCITY_THRESHOLD = 10.0; // calls per minute
    private static final double PREMIUM_RATE_THRESHOLD = 50.0; // cost threshold
    private static final double SIM_BOX_THRESHOLD = 100.0; // calls per hour
    private static final double ROAMING_COST_THRESHOLD = 200.0;
    private static final double ACCOUNT_TAKEOVER_THRESHOLD = 5.0; // location changes per hour
    private static final int TIME_WINDOW_MINUTES = 60;
    
    // State descriptors
    private transient ValueState<UserBehaviorProfile> userProfileState;
    private transient ValueState<CallHistory> callHistoryState;
    private transient ValueState<LocationHistory> locationHistoryState;
    
    // Risk scoring weights
    private static final Map<String, Double> RISK_WEIGHTS = new HashMap<>();
    static {
        RISK_WEIGHTS.put("velocity", 0.25);
        RISK_WEIGHTS.put("premium_rate", 0.20);
        RISK_WEIGHTS.put("sim_box", 0.30);
        RISK_WEIGHTS.put("roaming", 0.15);
        RISK_WEIGHTS.put("location_anomaly", 0.10);
    }
    
    // Machine learning model cache (simplified)
    private static final Map<String, Double> ML_PREDICTIONS = new ConcurrentHashMap<>();
    
    private ObjectMapper objectMapper;
    
    @Override
    public void open(Configuration parameters) throws Exception {
        // Initialize state descriptors
        userProfileState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("userProfile", UserBehaviorProfile.class)
        );
        
        callHistoryState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("callHistory", CallHistory.class)
        );
        
        locationHistoryState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("locationHistory", LocationHistory.class)
        );
        
        objectMapper = new ObjectMapper();
        
        logger.info("Advanced Fraud Detection Processor initialized");
    }
    
    @Override
    public void processElement(
            CDR cdr, 
            Context ctx, 
            Collector<FraudAlert> out
    ) throws Exception {
        
        String userId = cdr.getCallerId();
        long timestamp = ctx.timestamp();
        
        // Get or create user profile
        UserBehaviorProfile profile = userProfileState.value();
        if (profile == null) {
            profile = new UserBehaviorProfile(userId);
        }
        
        // Get call history
        CallHistory callHistory = callHistoryState.value();
        if (callHistory == null) {
            callHistory = new CallHistory();
        }
        
        // Get location history
        LocationHistory locationHistory = locationHistoryState.value();
        if (locationHistory == null) {
            locationHistory = new LocationHistory();
        }
        
        // Update histories
        callHistory.addCall(cdr, timestamp);
        locationHistory.addLocation(cdr.getCallerLocation(), timestamp);
        profile.updateProfile(cdr, timestamp);
        
        // Perform fraud detection analyses
        List<FraudDetectionResult> results = new ArrayList<>();
        
        // 1. Velocity Fraud Detection
        results.add(detectVelocityFraud(cdr, callHistory, timestamp));
        
        // 2. Premium Rate Fraud Detection
        results.add(detectPremiumRateFraud(cdr, callHistory, timestamp));
        
        // 3. SIM Box Fraud Detection
        results.add(detectSimBoxFraud(cdr, callHistory, profile, timestamp));
        
        // 4. Roaming Fraud Detection
        results.add(detectRoamingFraud(cdr, callHistory, timestamp));
        
        // 5. Account Takeover Detection
        results.add(detectAccountTakeover(cdr, locationHistory, profile, timestamp));
        
        // 6. Location Anomaly Detection
        results.add(detectLocationAnomaly(cdr, locationHistory, profile, timestamp));
        
        // Calculate composite risk score
        double riskScore = calculateCompositeRiskScore(results);
        
        // Generate alerts for high-risk activities
        for (FraudDetectionResult result : results) {
            if (result.isAnomalous() && result.getRiskScore() > 0.5) {
                FraudAlert alert = createFraudAlert(cdr, result, riskScore, timestamp);
                out.collect(alert);
                
                logger.warn("Fraud alert generated: {} for user {} with risk score {}",
                    result.getFraudType(), userId, result.getRiskScore());
            }
        }
        
        // Update states
        userProfileState.update(profile);
        callHistoryState.update(callHistory);
        locationHistoryState.update(locationHistory);
        
        // Clean up old data (sliding window)
        cleanupOldData(callHistory, locationHistory, timestamp);
    }
    
    /**
     * Detect velocity fraud (rapid successive calls)
     */
    private FraudDetectionResult detectVelocityFraud(
            CDR cdr, 
            CallHistory callHistory, 
            long timestamp
    ) {
        List<CDR> recentCalls = callHistory.getCallsInWindow(timestamp, TIME_WINDOW_MINUTES * 60 * 1000);
        
        // Group calls by minute to detect bursts
        Map<Long, Integer> callsPerMinute = new HashMap<>();
        for (CDR call : recentCalls) {
            long minute = call.getTimestamp() / (60 * 1000);
            callsPerMinute.put(minute, callsPerMinute.getOrDefault(minute, 0) + 1);
        }
        
        // Find maximum calls per minute
        int maxCallsPerMinute = callsPerMinute.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        
        boolean isAnomalous = maxCallsPerMinute > VELOCITY_THRESHOLD;
        double riskScore = Math.min(1.0, maxCallsPerMinute / (VELOCITY_THRESHOLD * 2));
        
        Map<String, Object> evidence = new HashMap<>();
        evidence.put("max_calls_per_minute", maxCallsPerMinute);
        evidence.put("threshold", VELOCITY_THRESHOLD);
        evidence.put("total_calls_in_window", recentCalls.size());
        
        return new FraudDetectionResult(
            "VELOCITY_FRAUD",
            isAnomalous,
            riskScore,
            String.format("High call velocity detected: %d calls per minute", maxCallsPerMinute),
            evidence
        );
    }
    
    /**
     * Detect premium rate fraud
     */
    private FraudDetectionResult detectPremiumRateFraud(
            CDR cdr, 
            CallHistory callHistory,
            long timestamp
    ) {
        // Check current call
        boolean currentCallPremium = cdr.getCost() > PREMIUM_RATE_THRESHOLD;
        
        // Check recent premium calls
        List<CDR> recentCalls = callHistory.getCallsInWindow(timestamp, TIME_WINDOW_MINUTES * 60 * 1000);
        long premiumCallsCount = recentCalls.stream()
            .filter(call -> call.getCost() > PREMIUM_RATE_THRESHOLD)
            .count();
        
        double totalPremiumCost = recentCalls.stream()
            .filter(call -> call.getCost() > PREMIUM_RATE_THRESHOLD)
            .mapToDouble(CDR::getCost)
            .sum();
        
        boolean isAnomalous = currentCallPremium && (premiumCallsCount > 3 || totalPremiumCost > 500.0);
        double riskScore = Math.min(1.0, (premiumCallsCount * 0.2) + (totalPremiumCost / 1000.0));
        
        Map<String, Object> evidence = new HashMap<>();
        evidence.put("current_call_cost", cdr.getCost());
        evidence.put("premium_calls_count", premiumCallsCount);
        evidence.put("total_premium_cost", totalPremiumCost);
        evidence.put("threshold", PREMIUM_RATE_THRESHOLD);
        
        return new FraudDetectionResult(
            "PREMIUM_RATE_FRAUD",
            isAnomalous,
            riskScore,
            String.format("Premium rate fraud detected: %.2f cost, %d premium calls", 
                totalPremiumCost, premiumCallsCount),
            evidence
        );
    }
    
    /**
     * Detect SIM box fraud (high volume automated calling)
     */
    private FraudDetectionResult detectSimBoxFraud(
            CDR cdr,
            CallHistory callHistory,
            UserBehaviorProfile profile,
            long timestamp
    ) {
        List<CDR> recentCalls = callHistory.getCallsInWindow(timestamp, TIME_WINDOW_MINUTES * 60 * 1000);
        
        // Analyze call patterns
        Set<String> uniqueDestinations = new HashSet<>();
        int shortCallsCount = 0;
        int internationalCallsCount = 0;
        double avgCallDuration = 0.0;
        
        for (CDR call : recentCalls) {
            uniqueDestinations.add(call.getCalleeId());
            if (call.getDuration() < 30) { // Less than 30 seconds
                shortCallsCount++;
            }
            if (call.getCallType().equals("INTERNATIONAL")) {
                internationalCallsCount++;
            }
            avgCallDuration += call.getDuration();
        }
        
        if (!recentCalls.isEmpty()) {
            avgCallDuration /= recentCalls.size();
        }
        
        // SIM box indicators
        boolean highVolume = recentCalls.size() > SIM_BOX_THRESHOLD;
        boolean manyDestinations = uniqueDestinations.size() > (recentCalls.size() * 0.8);
        boolean shortDurations = shortCallsCount > (recentCalls.size() * 0.6);
        boolean manyInternational = internationalCallsCount > (recentCalls.size() * 0.4);
        
        boolean isAnomalous = highVolume && (manyDestinations || shortDurations || manyInternational);
        
        double riskScore = 0.0;
        if (highVolume) riskScore += 0.4;
        if (manyDestinations) riskScore += 0.2;
        if (shortDurations) riskScore += 0.2;
        if (manyInternational) riskScore += 0.2;
        
        riskScore = Math.min(1.0, riskScore);
        
        Map<String, Object> evidence = new HashMap<>();
        evidence.put("calls_in_hour", recentCalls.size());
        evidence.put("unique_destinations", uniqueDestinations.size());
        evidence.put("short_calls_count", shortCallsCount);
        evidence.put("international_calls_count", internationalCallsCount);
        evidence.put("avg_call_duration", avgCallDuration);
        
        return new FraudDetectionResult(
            "SIM_BOX_FRAUD",
            isAnomalous,
            riskScore,
            String.format("SIM box pattern detected: %d calls, %d destinations", 
                recentCalls.size(), uniqueDestinations.size()),
            evidence
        );
    }
    
    /**
     * Detect roaming fraud
     */
    private FraudDetectionResult detectRoamingFraud(
            CDR cdr,
            CallHistory callHistory,
            long timestamp
    ) {
        if (!cdr.isRoamingFlag()) {
            return new FraudDetectionResult("ROAMING_FRAUD", false, 0.0, "Not a roaming call", new HashMap<>());
        }
        
        List<CDR> recentRoamingCalls = callHistory.getCallsInWindow(timestamp, TIME_WINDOW_MINUTES * 60 * 1000)
            .stream()
            .filter(CDR::isRoamingFlag)
            .collect(java.util.stream.Collectors.toList());
        
        double totalRoamingCost = recentRoamingCalls.stream()
            .mapToDouble(CDR::getCost)
            .sum();
        
        // Check for sudden spike in roaming activity
        boolean suddenRoamingSpike = recentRoamingCalls.size() > 20; // More than 20 roaming calls per hour
        boolean highRoamingCost = totalRoamingCost > ROAMING_COST_THRESHOLD;
        
        boolean isAnomalous = suddenRoamingSpike || highRoamingCost;
        double riskScore = Math.min(1.0, 
            (recentRoamingCalls.size() / 50.0) + (totalRoamingCost / 500.0));
        
        Map<String, Object> evidence = new HashMap<>();
        evidence.put("roaming_calls_count", recentRoamingCalls.size());
        evidence.put("total_roaming_cost", totalRoamingCost);
        evidence.put("current_location", cdr.getCallerLocation());
        evidence.put("cost_threshold", ROAMING_COST_THRESHOLD);
        
        return new FraudDetectionResult(
            "ROAMING_FRAUD",
            isAnomalous,
            riskScore,
            String.format("Roaming fraud detected: %d calls, %.2f cost", 
                recentRoamingCalls.size(), totalRoamingCost),
            evidence
        );
    }
    
    /**
     * Detect account takeover based on location changes
     */
    private FraudDetectionResult detectAccountTakeover(
            CDR cdr,
            LocationHistory locationHistory,
            UserBehaviorProfile profile,
            long timestamp
    ) {
        List<String> recentLocations = locationHistory.getLocationsInWindow(
            timestamp, TIME_WINDOW_MINUTES * 60 * 1000);
        
        Set<String> uniqueLocations = new HashSet<>(recentLocations);
        
        // Check for impossible travel (multiple distant locations in short time)
        boolean rapidLocationChanges = uniqueLocations.size() > ACCOUNT_TAKEOVER_THRESHOLD;
        
        // Check against user's normal location pattern
        Set<String> normalLocations = profile.getFrequentLocations();
        long unfamiliarLocationCount = uniqueLocations.stream()
            .filter(loc -> !normalLocations.contains(loc))
            .count();
        
        boolean unfamiliarLocations = unfamiliarLocationCount > (uniqueLocations.size() * 0.7);
        
        boolean isAnomalous = rapidLocationChanges || unfamiliarLocations;
        double riskScore = Math.min(1.0, 
            (uniqueLocations.size() / 10.0) + (unfamiliarLocationCount / 5.0));
        
        Map<String, Object> evidence = new HashMap<>();
        evidence.put("unique_locations_count", uniqueLocations.size());
        evidence.put("unfamiliar_locations_count", unfamiliarLocationCount);
        evidence.put("current_location", cdr.getCallerLocation());
        evidence.put("frequent_locations", profile.getFrequentLocations());
        
        return new FraudDetectionResult(
            "ACCOUNT_TAKEOVER",
            isAnomalous,
            riskScore,
            String.format("Potential account takeover: %d locations, %d unfamiliar", 
                uniqueLocations.size(), unfamiliarLocationCount),
            evidence
        );
    }
    
    /**
     * Detect location anomalies
     */
    private FraudDetectionResult detectLocationAnomaly(
            CDR cdr,
            LocationHistory locationHistory,
            UserBehaviorProfile profile,
            long timestamp
    ) {
        String currentLocation = cdr.getCallerLocation();
        Set<String> normalLocations = profile.getFrequentLocations();
        
        // Check if current location is completely new
        boolean newLocation = !normalLocations.contains(currentLocation);
        
        // Check distance from normal locations (simplified)
        boolean distantLocation = isLocationDistant(currentLocation, normalLocations);
        
        // Check time of day anomaly
        LocalDateTime dateTime = LocalDateTime.ofInstant(
            Instant.ofEpochMilli(timestamp), ZoneOffset.UTC);
        int hour = dateTime.getHour();
        boolean unusualTime = hour < 6 || hour > 22; // Very early or very late
        
        boolean isAnomalous = newLocation && (distantLocation || unusualTime);
        double riskScore = 0.0;
        if (newLocation) riskScore += 0.3;
        if (distantLocation) riskScore += 0.4;
        if (unusualTime) riskScore += 0.3;
        
        riskScore = Math.min(1.0, riskScore);
        
        Map<String, Object> evidence = new HashMap<>();
        evidence.put("current_location", currentLocation);
        evidence.put("is_new_location", newLocation);
        evidence.put("is_distant_location", distantLocation);
        evidence.put("call_hour", hour);
        evidence.put("frequent_locations", normalLocations);
        
        return new FraudDetectionResult(
            "LOCATION_ANOMALY",
            isAnomalous,
            riskScore,
            String.format("Location anomaly detected at %s during hour %d", 
                currentLocation, hour),
            evidence
        );
    }
    
    /**
     * Calculate composite risk score from all detection results
     */
    private double calculateCompositeRiskScore(List<FraudDetectionResult> results) {
        double weightedScore = 0.0;
        double totalWeight = 0.0;
        
        for (FraudDetectionResult result : results) {
            String fraudType = result.getFraudType().toLowerCase();
            String key = fraudType.contains("velocity") ? "velocity" :
                        fraudType.contains("premium") ? "premium_rate" :
                        fraudType.contains("sim") ? "sim_box" :
                        fraudType.contains("roaming") ? "roaming" :
                        "location_anomaly";
            
            Double weight = RISK_WEIGHTS.get(key);
            if (weight != null) {
                weightedScore += result.getRiskScore() * weight;
                totalWeight += weight;
            }
        }
        
        return totalWeight > 0 ? weightedScore / totalWeight : 0.0;
    }
    
    /**
     * Create fraud alert from detection result
     */
    private FraudAlert createFraudAlert(
            CDR cdr, 
            FraudDetectionResult result, 
            double compositeRiskScore,
            long timestamp
    ) {
        FraudAlert alert = new FraudAlert();
        alert.setAlertId(UUID.randomUUID().toString());
        alert.setUserId(cdr.getCallerId());
        alert.setFraudType(result.getFraudType());
        alert.setRiskScore(result.getRiskScore());
        alert.setCompositeRiskScore(compositeRiskScore);
        alert.setDescription(result.getDescription());
        alert.setEvidence(result.getEvidence());
        alert.setTimestamp(timestamp);
        alert.setCallId(cdr.getCallId());
        alert.setLocation(cdr.getCallerLocation());
        
        // Determine severity based on risk score
        if (compositeRiskScore >= 0.8) {
            alert.setSeverity("CRITICAL");
        } else if (compositeRiskScore >= 0.6) {
            alert.setSeverity("HIGH");
        } else if (compositeRiskScore >= 0.4) {
            alert.setSeverity("MEDIUM");
        } else {
            alert.setSeverity("LOW");
        }
        
        return alert;
    }
    
    /**
     * Clean up old data from sliding windows
     */
    private void cleanupOldData(
            CallHistory callHistory, 
            LocationHistory locationHistory, 
            long currentTimestamp
    ) {
        long cutoffTime = currentTimestamp - (TIME_WINDOW_MINUTES * 60 * 1000L * 2); // Keep 2x window size
        callHistory.removeOldCalls(cutoffTime);
        locationHistory.removeOldLocations(cutoffTime);
    }
    
    /**
     * Simplified distance check for locations
     */
    private boolean isLocationDistant(String currentLocation, Set<String> normalLocations) {
        // In a real implementation, this would use actual geolocation
        // For now, we use a simple string-based heuristic
        if (normalLocations.isEmpty()) {
            return false;
        }
        
        for (String normalLoc : normalLocations) {
            if (currentLocation.startsWith(normalLoc.substring(0, Math.min(3, normalLoc.length())))) {
                return false; // Same area/region
            }
        }
        
        return true; // Potentially distant
    }
    
    /**
     * Inner class for fraud detection results
     */
    private static class FraudDetectionResult {
        private final String fraudType;
        private final boolean anomalous;
        private final double riskScore;
        private final String description;
        private final Map<String, Object> evidence;
        
        public FraudDetectionResult(String fraudType, boolean anomalous, double riskScore, 
                                  String description, Map<String, Object> evidence) {
            this.fraudType = fraudType;
            this.anomalous = anomalous;
            this.riskScore = riskScore;
            this.description = description;
            this.evidence = evidence;
        }
        
        public String getFraudType() { return fraudType; }
        public boolean isAnomalous() { return anomalous; }
        public double getRiskScore() { return riskScore; }
        public String getDescription() { return description; }
        public Map<String, Object> getEvidence() { return evidence; }
    }
}