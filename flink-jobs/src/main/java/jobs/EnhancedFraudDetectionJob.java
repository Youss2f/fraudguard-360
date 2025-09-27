package jobs;

import models.CDR;
import models.FraudAlert;
import models.FraudStatistics;
import processors.AdvancedFraudDetectionProcessor;
import aggregators.FraudStatisticsAggregator;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;

/**
 * Enhanced Fraud Detection Job with Real-time Processing
 * Processes CDR data from Kafka and generates fraud alerts
 */
public class EnhancedFraudDetectionJob {
    
    private static final Logger logger = LoggerFactory.getLogger(EnhancedFraudDetectionJob.class);
    
    public static void main(String[] args) throws Exception {
        // Set up the streaming execution environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // Configure for production
        env.setParallelism(4); // Adjust based on your cluster
        env.enableCheckpointing(30000); // Checkpoint every 30 seconds
        
        logger.info("Starting Enhanced Fraud Detection Job");
        
        try {
            // Create test CDR data source (replace with Kafka in production)
            DataStream<CDR> cdrStream = env
                .addSource(new sources.TestCDRSource())
                .assignTimestampsAndWatermarks(
                    WatermarkStrategy.<CDR>forBoundedOutOfOrderness(Duration.ofSeconds(20))
                        .withTimestampAssigner((cdr, timestamp) -> cdr.getEventTimestamp()))
                .name("CDR Input Stream");
            
            // Apply fraud detection processing
            DataStream<FraudAlert> fraudAlerts = cdrStream
                .keyBy(CDR::getCallerId)
                .process(new AdvancedFraudDetectionProcessor())
                .name("Fraud Detection Processor");
            
            // Create high-priority alerts stream (critical and high severity)
            DataStream<FraudAlert> highPriorityAlerts = fraudAlerts
                .filter(alert -> "CRITICAL".equals(alert.getSeverity()) || 
                               "HIGH".equals(alert.getSeverity()))
                .name("High Priority Alerts");
            
            // Create aggregated fraud statistics
            DataStream<FraudStatistics> fraudStats = fraudAlerts
                .windowAll(TumblingEventTimeWindows.of(Time.minutes(5)))
                .aggregate(new FraudStatisticsAggregator())
                .name("Fraud Statistics Aggregator");
            
            // Set up output sinks
            setupOutputSinks(fraudAlerts, highPriorityAlerts, fraudStats);
            
            // Execute the job
            env.execute("Enhanced Fraud Detection Job");
            
        } catch (Exception e) {
            logger.error("Error in Enhanced Fraud Detection Job", e);
            throw e;
        }
    }
    
    /**
     * Set up output sinks for fraud alerts and statistics
     */
    private static void setupOutputSinks(
            DataStream<FraudAlert> fraudAlerts,
            DataStream<FraudAlert> highPriorityAlerts,
            DataStream<FraudStatistics> fraudStats
    ) {
        
        // Console output for fraud alerts
        fraudAlerts.print("FRAUD_ALERT").name("Fraud Alerts Console Output");
        
        // Console output for high priority alerts
        highPriorityAlerts.print("HIGH_PRIORITY_ALERT").name("High Priority Alerts Console Output");
        
        // Console output for fraud statistics
        fraudStats.print("FRAUD_STATISTICS").name("Fraud Statistics Console Output");
        
        logger.info("Output sinks configured successfully (using console output)");
    }
}