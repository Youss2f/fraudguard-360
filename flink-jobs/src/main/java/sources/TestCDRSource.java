package sources;

import models.CDR;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.Random;

/**
 * Test CDR Source for generating sample data
 */
public class TestCDRSource implements SourceFunction<CDR> {
    
    private static final Logger logger = LoggerFactory.getLogger(TestCDRSource.class);
    private volatile boolean isRunning = true;
    private final Random random = new Random();
    
    // Test data arrays
    private static final String[] USER_IDS = {
        "user001", "user002", "user003", "user004", "user005",
        "user006", "user007", "user008", "user009", "user010"
    };
    
    private static final String[] LOCATIONS = {
        "New York", "London", "Tokyo", "Singapore", "Berlin",
        "Paris", "Sydney", "Mumbai", "Shanghai", "Toronto"
    };
    
    private static final String[] CALL_TYPES = {
        "LOCAL", "NATIONAL", "INTERNATIONAL", "ROAMING"
    };
    
    @Override
    public void run(SourceContext<CDR> ctx) throws Exception {
        logger.info("Starting Test CDR Source");
        
        int cdrCount = 0;
        
        while (isRunning && cdrCount < 10000) { // Generate 10k test records
            CDR cdr = generateTestCDR(cdrCount);
            ctx.collect(cdr);
            cdrCount++;
            
            // Add some delay to simulate real-time streaming
            Thread.sleep(random.nextInt(100) + 10); // 10-110ms delay
            
            if (cdrCount % 1000 == 0) {
                logger.info("Generated {} CDR records", cdrCount);
            }
        }
        
        logger.info("Test CDR Source completed. Generated {} records", cdrCount);
    }
    
    @Override
    public void cancel() {
        isRunning = false;
        logger.info("Test CDR Source cancelled");
    }
    
    /**
     * Generate a test CDR with realistic fraud patterns
     */
    private CDR generateTestCDR(int id) {
        CDR cdr = new CDR();
        
        // Basic fields
        cdr.setId("cdr_" + String.format("%06d", id));
        cdr.setCallerId(USER_IDS[random.nextInt(USER_IDS.length)]);
        cdr.setCalleeId(generateCalleeId());
        cdr.setCallType(CALL_TYPES[random.nextInt(CALL_TYPES.length)]);
        
        // Time fields
        long now = System.currentTimeMillis();
        long startTime = now - random.nextInt(3600000); // Within last hour
        cdr.setStartTime(Instant.ofEpochMilli(startTime).toString());
        
        int duration = generateDuration();
        cdr.setDuration(duration);
        cdr.setEndTime(Instant.ofEpochMilli(startTime + duration * 1000L).toString());
        
        // Location and network
        cdr.setLocationCaller(LOCATIONS[random.nextInt(LOCATIONS.length)]);
        cdr.setLocationCallee(LOCATIONS[random.nextInt(LOCATIONS.length)]);
        cdr.setTowerId("tower_" + random.nextInt(100));
        cdr.setDeviceImei(generateIMEI());
        
        // Cost calculation
        double cost = calculateCost(cdr.getCallType(), duration);
        cdr.setCost(cost);
        
        // Country code
        cdr.setCountryCode(generateCountryCode());
        
        // Bytes transmitted (for data calls)
        cdr.setBytesTransmitted((long) (random.nextDouble() * 1000000)); // Up to 1MB
        
        // Inject some fraud patterns
        injectFraudPatterns(cdr, id);
        
        return cdr;
    }
    
    /**
     * Generate a callee ID (destination number)
     */
    private String generateCalleeId() {
        if (random.nextDouble() < 0.1) { // 10% premium numbers
            return "+19001234" + String.format("%03d", random.nextInt(1000));
        } else if (random.nextDouble() < 0.2) { // 20% international
            return "+" + (random.nextInt(99) + 1) + "123456" + String.format("%04d", random.nextInt(10000));
        } else { // 70% regular numbers
            return "+1555" + String.format("%07d", random.nextInt(10000000));
        }
    }
    
    /**
     * Generate call duration with realistic distribution
     */
    private int generateDuration() {
        double rand = random.nextDouble();
        
        if (rand < 0.1) { // 10% very short calls (potential fraud)
            return random.nextInt(30);
        } else if (rand < 0.3) { // 20% short calls
            return 30 + random.nextInt(60);
        } else if (rand < 0.8) { // 50% normal calls
            return 90 + random.nextInt(300);
        } else { // 20% long calls
            return 400 + random.nextInt(1800);
        }
    }
    
    /**
     * Calculate call cost based on type and duration
     */
    private double calculateCost(String callType, int duration) {
        double ratePerMinute;
        
        switch (callType) {
            case "LOCAL":
                ratePerMinute = 0.05;
                break;
            case "NATIONAL":
                ratePerMinute = 0.10;
                break;
            case "INTERNATIONAL":
                ratePerMinute = 0.50;
                break;
            case "ROAMING":
                ratePerMinute = 1.00;
                break;
            default:
                ratePerMinute = 0.05;
        }
        
        double minutes = duration / 60.0;
        double baseCost = ratePerMinute * minutes;
        
        // Add some randomness
        return baseCost * (0.8 + random.nextDouble() * 0.4); // ±20% variation
    }
    
    /**
     * Generate IMEI number
     */
    private String generateIMEI() {
        return String.format("%015d", Math.abs(random.nextLong()) % 1000000000000000L);
    }
    
    /**
     * Generate country code
     */
    private String generateCountryCode() {
        String[] countryCodes = {"US", "UK", "JP", "SG", "DE", "FR", "AU", "IN", "CN", "CA"};
        return countryCodes[random.nextInt(countryCodes.length)];
    }
    
    /**
     * Inject fraud patterns into some CDRs for testing
     */
    private void injectFraudPatterns(CDR cdr, int id) {
        double fraudProbability = 0.05; // 5% fraud rate
        
        if (random.nextDouble() < fraudProbability) {
            String fraudType = generateFraudType();
            
            switch (fraudType) {
                case "VELOCITY_FRAUD":
                    // High volume calling pattern (will be detected by processor)
                    if (cdr.getCallerId().equals("user001")) {
                        cdr.setCallType("LOCAL");
                        cdr.setDuration(random.nextInt(30)); // Short calls
                    }
                    break;
                    
                case "PREMIUM_RATE_FRAUD":
                    cdr.setCalleeId("+19001234567"); // Premium rate number
                    cdr.setCost(100.0 + random.nextDouble() * 200.0); // High cost
                    cdr.setCallType("INTERNATIONAL");
                    break;
                    
                case "SIM_BOX_FRAUD":
                    if (cdr.getCallerId().equals("user002")) {
                        cdr.setCallType("INTERNATIONAL");
                        cdr.setDuration(random.nextInt(60)); // Short international calls
                        cdr.setCost(random.nextDouble() * 10.0); // Suspiciously low cost
                    }
                    break;
                    
                case "ROAMING_FRAUD":
                    cdr.setCallType("ROAMING");
                    cdr.setCost(50.0 + random.nextDouble() * 100.0); // High roaming cost
                    cdr.setLocationCaller("Remote Location");
                    break;
                    
                case "ACCOUNT_TAKEOVER":
                    if (cdr.getCallerId().equals("user003")) {
                        // Multiple locations for same user (processor will detect pattern)
                        cdr.setLocationCaller(LOCATIONS[random.nextInt(LOCATIONS.length)]);
                    }
                    break;
            }
        }
    }
    
    /**
     * Generate random fraud type for testing
     */
    private String generateFraudType() {
        String[] fraudTypes = {
            "VELOCITY_FRAUD", "PREMIUM_RATE_FRAUD", "SIM_BOX_FRAUD", 
            "ROAMING_FRAUD", "ACCOUNT_TAKEOVER"
        };
        return fraudTypes[random.nextInt(fraudTypes.length)];
    }
}