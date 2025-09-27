package processors;

import models.CDR;
import models.UserFeatures;
import models.FraudAlert;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

/**
 * Main fraud detection processor that analyzes CDR events in real-time
 * using windowed aggregation and rule-based detection.
 */
public class FraudDetectionProcessor extends KeyedProcessFunction<String, CDR, FraudAlert> {
    
    private static final Logger logger = LoggerFactory.getLogger(FraudDetectionProcessor.class);
    
    // Time windows in milliseconds
    private static final long WINDOW_SIZE_5MIN = 5 * 60 * 1000L;
    private static final long WINDOW_SIZE_30MIN = 30 * 60 * 1000L;
    private static final long WINDOW_SIZE_1HOUR = 60 * 60 * 1000L;
    
    // Enhanced risk thresholds
    private static final double MEDIUM_RISK_THRESHOLD = 0.5;
    private static final double HIGH_RISK_THRESHOLD = 0.7;
    private static final double CRITICAL_RISK_THRESHOLD = 0.9;
    
    // Fraud pattern thresholds
    private static final int MAX_CALLS_PER_5MIN = 100;
    private static final int MAX_UNIQUE_CALLEES_PER_HOUR = 50;
    private static final double MAX_AVERAGE_CALL_DURATION = 300.0; // 5 minutes
    private static final int MIN_NIGHT_HOURS_START = 23;
    private static final int MIN_NIGHT_HOURS_END = 6;
    
    // State to maintain user features
    private transient ValueState<UserFeatures> userFeaturesState;
    private transient ValueState<Set<String>> uniqueCalleesState;
    
    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        
        // Initialize state descriptors
        ValueStateDescriptor<UserFeatures> featuresDescriptor = 
            new ValueStateDescriptor<>("userFeatures", TypeInformation.of(UserFeatures.class));
        userFeaturesState = getRuntimeContext().getState(featuresDescriptor);
        
        ValueStateDescriptor<Set<String>> calleesDescriptor = 
            new ValueStateDescriptor<>("uniqueCallees", TypeInformation.of((Class<Set<String>>) (Class<?>) Set.class));
        uniqueCalleesState = getRuntimeContext().getState(calleesDescriptor);
    }

    @Override
    public void processElement(CDR cdr, Context context, Collector<FraudAlert> collector) throws Exception {
        
        long eventTime = cdr.getEventTimestamp();
        String userId = cdr.getCallerId();
        
        logger.debug("Processing CDR for user: {} at time: {}", userId, eventTime);
        
        // Get or create user features for current window
        UserFeatures features = userFeaturesState.value();
        if (features == null || shouldResetWindow(features, eventTime)) {
            features = new UserFeatures(userId, eventTime, eventTime + WINDOW_SIZE_30MIN);
            uniqueCalleesState.update(new HashSet<>());
        }
        
        // Update features with current CDR
        updateFeatures(features, cdr);
        
        // Update unique callees
        Set<String> uniqueCallees = uniqueCalleesState.value();
        if (uniqueCallees == null) {
            uniqueCallees = new HashSet<>();
        }
        uniqueCallees.add(cdr.getCalleeId());
        features.setUniqueCallees(uniqueCallees.size());
        
        // Calculate derived features
        features.calculateDerivedFeatures();
        
        // Save updated state
        userFeaturesState.update(features);
        uniqueCalleesState.update(uniqueCallees);
        
        // Check for fraud patterns and generate alerts
        checkFraudPatterns(features, cdr, collector);
        
        // Set timer for window cleanup
        long windowEnd = features.getWindowEnd();
        context.timerService().registerEventTimeTimer(windowEnd);
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<FraudAlert> out) throws Exception {
        // Clean up expired state
        UserFeatures features = userFeaturesState.value();
        if (features != null && timestamp >= features.getWindowEnd()) {
            logger.debug("Cleaning up expired window for user: {}", features.getUserId());
            userFeaturesState.clear();
            uniqueCalleesState.clear();
        }
    }

    private boolean shouldResetWindow(UserFeatures features, long eventTime) {
        return eventTime >= features.getWindowEnd();
    }

    private void updateFeatures(UserFeatures features, CDR cdr) {
        // Update call counts by type
        switch (cdr.getCallType() != null ? cdr.getCallType().toLowerCase() : "voice") {
            case "voice":
                features.setCallCount(features.getCallCount() + 1);
                if (cdr.getDuration() != null) {
                    features.setTotalDuration(features.getTotalDuration() + cdr.getDuration());
                }
                break;
            case "sms":
                features.setSmsCount(features.getSmsCount() + 1);
                break;
            case "data":
                features.setDataSessionCount(features.getDataSessionCount() + 1);
                break;
        }
        
        // Update cost
        if (cdr.getCost() != null) {
            features.setTotalCost(features.getTotalCost() + cdr.getCost());
        }
        
        // Check for international calls
        if (isInternationalCall(cdr)) {
            features.setInternationalCalls(features.getInternationalCalls() + 1);
        }
        
        // Check for premium rate calls
        if (isPremiumRateCall(cdr)) {
            features.setPremiumRateCalls(features.getPremiumRateCalls() + 1);
        }
        
        // Check for night calls
        if (isNightCall(cdr)) {
            features.setNightCalls(features.getNightCalls() + 1);
        }
        
        // Update location counts
        if (cdr.getLocationCaller() != null) {
            features.getLocationCounts().merge(cdr.getLocationCaller(), 1, Integer::sum);
        }
        
        // Update device counts
        if (cdr.getDeviceImei() != null) {
            features.getDeviceCounts().merge(cdr.getDeviceImei(), 1, Integer::sum);
        }
        
        // Update window end time
        features.setWindowEnd(Math.max(features.getWindowEnd(), cdr.getEventTimestamp() + WINDOW_SIZE_30MIN));
    }

    private void checkFraudPatterns(UserFeatures features, CDR cdr, Collector<FraudAlert> collector) {
        double riskScore = features.calculateRiskScore();
        
        // Generate alerts based on risk score
        if (riskScore >= CRITICAL_RISK_THRESHOLD) {
            generateAlert(features, cdr, "critical", riskScore, "CRITICAL fraud risk detected", collector);
        } else if (riskScore >= HIGH_RISK_THRESHOLD) {
            generateAlert(features, cdr, "high", riskScore, "HIGH fraud risk detected", collector);
        } else if (riskScore >= MEDIUM_RISK_THRESHOLD) {
            generateAlert(features, cdr, "medium", riskScore, "MEDIUM fraud risk detected", collector);
        }
        
        // Enhanced specific pattern detection
        checkSimBoxPattern(features, cdr, collector);
        checkVelocityPattern(features, cdr, collector);
        checkAnomalousLocationPattern(features, cdr, collector);
        checkAccountTakeoverPattern(features, cdr, collector);
        checkPremiumRatePattern(features, cdr, collector);
        checkRoamingFraudPattern(features, cdr, collector);
    }

    private void checkSimBoxPattern(UserFeatures features, CDR cdr, Collector<FraudAlert> collector) {
        // SIM box pattern: High volume of short international calls
        if (features.getInternationalCalls() > 50 && 
            features.getAvgCallDuration() < 30 && 
            features.getCallFrequency() > 15) {
            
            generateAlert(features, cdr, "critical", 0.95, 
                "SIM_BOX fraud pattern detected: High volume international calls", collector);
        }
    }

    private void checkVelocityPattern(UserFeatures features, CDR cdr, Collector<FraudAlert> collector) {
        // Velocity pattern: Too many calls in a short time
        if (features.getCallFrequency() > 30) {
            generateAlert(features, cdr, "high", 0.8, 
                "VELOCITY fraud pattern detected: Unusually high call frequency", collector);
        }
    }

    private void checkAnomalousLocationPattern(UserFeatures features, CDR cdr, Collector<FraudAlert> collector) {
        // Location anomaly: Calls from too many different locations
        if (features.getLocationCounts().size() > 10) {
            generateAlert(features, cdr, "medium", 0.6, 
                "LOCATION_ANOMALY: Calls from multiple suspicious locations", collector);
        }
    }

    private void generateAlert(UserFeatures features, CDR cdr, String severity, 
                             double riskScore, String description, Collector<FraudAlert> collector) {
        
        FraudAlert alert = new FraudAlert();
        alert.setAlertId(UUID.randomUUID().toString());
        alert.setUserId(features.getUserId());
        alert.setSeverity(severity);
        alert.setRiskScore(riskScore);
        alert.setConfidenceScore(Math.min(riskScore + 0.1, 1.0));
        alert.setDescription(description);
        alert.setFraudType(determineFraudType(description));
        
        // Add evidence
        alert.getEvidence().put("call_frequency", features.getCallFrequency());
        alert.getEvidence().put("international_calls", features.getInternationalCalls());
        alert.getEvidence().put("premium_rate_calls", features.getPremiumRateCalls());
        alert.getEvidence().put("night_calls", features.getNightCalls());
        alert.getEvidence().put("unique_locations", features.getLocationCounts().size());
        alert.getEvidence().put("unique_devices", features.getDeviceCounts().size());
        alert.getEvidence().put("triggering_cdr_id", cdr.getId());
        
        logger.info("Generated fraud alert: {} for user: {} with risk score: {}", 
                   alert.getAlertId(), features.getUserId(), riskScore);
        
        collector.collect(alert);
    }

    private String determineFraudType(String description) {
        if (description.contains("SIM_BOX")) return "sim_box";
        if (description.contains("VELOCITY")) return "subscription_fraud";
        if (description.contains("LOCATION")) return "roaming_fraud";
        return "suspicious_activity";
    }

    private boolean isInternationalCall(CDR cdr) {
        return cdr.getCountryCode() != null && !cdr.getCountryCode().equals("US");
    }

    private boolean isPremiumRateCall(CDR cdr) {
        String calleeId = cdr.getCalleeId();
        return calleeId != null && (calleeId.startsWith("900") || calleeId.startsWith("976"));
    }

    private boolean isNightCall(CDR cdr) {
        try {
            Instant instant = Instant.parse(cdr.getStartTime());
            LocalTime time = instant.atZone(ZoneId.systemDefault()).toLocalTime();
            return time.isBefore(LocalTime.of(6, 0)) || time.isAfter(LocalTime.of(22, 0));
        } catch (Exception e) {
            return false;
        }
    }
    
    private void checkAccountTakeoverPattern(UserFeatures features, CDR cdr, Collector<FraudAlert> collector) {
        // Account takeover: Sudden change in behavior patterns
        if (features.getLocationCounts().size() > 3 && 
            features.getDeviceCounts().size() > 2 &&
            features.getCallFrequency() > 20) {
            
            generateAlert(features, cdr, "high", 0.85, 
                "ACCOUNT_TAKEOVER pattern detected: Multiple devices and locations", collector);
        }
    }
    
    private void checkPremiumRatePattern(UserFeatures features, CDR cdr, Collector<FraudAlert> collector) {
        // Premium rate fraud: Excessive premium rate calls
        if (features.getPremiumRateCalls() > 10 || 
            (features.getPremiumRateCalls() > 0 && features.getCallCount() > 0 && 
             (double)features.getPremiumRateCalls() / features.getCallCount() > 0.3)) {
            
            generateAlert(features, cdr, "critical", 0.92, 
                "PREMIUM_RATE fraud pattern detected: Excessive premium calls", collector);
        }
    }
    
    private void checkRoamingFraudPattern(UserFeatures features, CDR cdr, Collector<FraudAlert> collector) {
        // Roaming fraud: High international activity with suspicious patterns
        if (features.getInternationalCalls() > 30 && 
            features.getNightCalls() > 15 &&
            features.getLocationCounts().size() > 5) {
            
            generateAlert(features, cdr, "high", 0.80, 
                "ROAMING_FRAUD pattern detected: Suspicious international activity", collector);
        }
    }
}