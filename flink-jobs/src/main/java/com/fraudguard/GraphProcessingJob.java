package com.fraudguard;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.streaming.util.serialization.SimpleStringSchema;
import com.fasterxml.jackson.databind.ObjectMapper;
import models.CDR;
import models.FraudAlert;
import processors.FraudDetectionProcessor;
import sinks.CDRNeo4jSink;
import sinks.AlertNeo4jSink;
import sinks.AlertSink;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Main Flink job for real-time fraud detection in telecom CDR data
 */
public class GraphProcessingJob {
    private static final Logger logger = LoggerFactory.getLogger(GraphProcessingJob.class);
    
    private static final String KAFKA_BOOTSTRAP_SERVERS = "kafka:9092";
    private static final String CDR_TOPIC = "telecom-cdr-topic";
    private static final String ALERTS_TOPIC = "fraud-alerts-topic";

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // Configure checkpointing for fault tolerance
        env.enableCheckpointing(60000);  // Checkpoint every minute
        
        // Set parallelism
        env.setParallelism(4);
        
        logger.info("Starting FraudGuard 360 Streaming Job");
        
        // Configure Kafka consumer
        Properties kafkaProps = new Properties();
        kafkaProps.setProperty("bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS);
        kafkaProps.setProperty("group.id", "fraudguard-consumer");
        kafkaProps.setProperty("auto.offset.reset", "latest");
        
        // Create Kafka source for CDR data
        DataStream<String> cdrJsonStream;
        
        try {
            FlinkKafkaConsumer<String> kafkaConsumer = new FlinkKafkaConsumer<>(
                CDR_TOPIC, 
                new SimpleStringSchema(), 
                kafkaProps
            );
            
            cdrJsonStream = env.addSource(kafkaConsumer).name("CDR-Kafka-Source");
            logger.info("Connected to Kafka topic: {}", CDR_TOPIC);
            
        } catch (Exception e) {
            logger.warn("Failed to connect to Kafka, using test data: {}", e.getMessage());
            // Fallback to test data for development
            cdrJsonStream = env.addSource(new TestCDRSource()).name("Test-CDR-Source");
        }
        
        // Parse JSON CDR data
        DataStream<CDR> cdrStream = cdrJsonStream
            .map(json -> {
                try {
                    ObjectMapper mapper = new ObjectMapper();
                    return mapper.readValue(json, CDR.class);
                } catch (Exception e) {
                    logger.error("Failed to parse CDR JSON: {}", json, e);
                    return null;
                }
            })
            .filter(cdr -> cdr != null)
            .name("CDR-JSON-Parser");
        
        // Key by user ID and process through fraud detection
        DataStream<FraudAlert> alertStream = cdrStream
            .keyBy(CDR::getCallerId)
            .process(new FraudDetectionProcessor())
            .name("Fraud-Detection-Processor");
        
        // Output fraud alerts to multiple sinks
        alertStream.addSink(new AlertSink(KAFKA_BOOTSTRAP_SERVERS, ALERTS_TOPIC))
                  .name("Alert-Kafka-Sink");
        
        alertStream.addSink(new AlertNeo4jSink("bolt://neo4j:7687", "neo4j", "password"))
                  .name("Alert-Neo4j-Sink");
        
        // Also store CDR data in Neo4j for graph analysis
        cdrStream.addSink(new CDRNeo4jSink("bolt://neo4j:7687", "neo4j", "password"))
                 .name("CDR-Neo4j-Sink");
        
        // Print alerts to console for monitoring
        alertStream.print("FRAUD-ALERT");
        
        logger.info("Job topology configured, starting execution...");
        env.execute("FraudGuard 360 Real-Time Fraud Detection");
    }
    
    /**
     * Test data source for development when Kafka is not available
     */
    private static class TestCDRSource implements SourceFunction<String> {
        private volatile boolean isRunning = true;
        private final ObjectMapper mapper = new ObjectMapper();
        
        @Override
        public void run(SourceContext<String> ctx) throws Exception {
            String[] testUsers = {"user_001", "user_002", "user_003", "user_004"};
            String[] callees = {"callee_001", "callee_002", "callee_003", "callee_004", "callee_005"};
            String[] countries = {"US", "CA", "UK", "DE", "FR"};
            
            while (isRunning) {
                for (String user : testUsers) {
                    CDR cdr = new CDR();
                    cdr.setId(java.util.UUID.randomUUID().toString());
                    cdr.setCallerId(user);
                    cdr.setCalleeId(callees[(int)(Math.random() * callees.length)]);
                    cdr.setCallType("voice");
                    cdr.setStartTime(java.time.Instant.now().toString());
                    cdr.setDuration((int)(Math.random() * 300) + 30);
                    cdr.setCost(Math.random() * 5.0);
                    cdr.setCountryCode(countries[(int)(Math.random() * countries.length)]);
                    cdr.setLocationCaller("Location_" + (int)(Math.random() * 10));
                    cdr.setDeviceImei("IMEI_" + user + "_" + (int)(Math.random() * 3));
                    
                    String json = mapper.writeValueAsString(cdr);
                    ctx.collect(json);
                }
                
                Thread.sleep(1000); // Generate test data every second
            }
        }
        
        @Override
        public void cancel() {
            isRunning = false;
        }
    }
}
