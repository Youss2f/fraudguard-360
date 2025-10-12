import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.test.util.MiniClusterWithClientResource;
import org.apache.flink.configuration.Configuration;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.neo4j.driver.Driver;
import org.neo4j.driver.Session;

import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Comprehensive unit tests for Flink stream processing jobs
 * Tests CDR processing, fraud detection, windowing operations, and sinks
 */
public class FlinkJobsTest {

    private StreamExecutionEnvironment env;
    private static final String KAFKA_BOOTSTRAP_SERVERS = "localhost:9092";
    private static final String NEO4J_URI = "bolt://localhost:7687";

    @Mock
    private Driver neo4jDriver;
    
    @Mock
    private Session neo4jSession;

    private final BlockingQueue<CDRData> processedCDRs = new LinkedBlockingQueue<>();
    private final BlockingQueue<FraudAlert> fraudAlerts = new LinkedBlockingQueue<>();

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
        env.setParallelism(1); // Use parallelism of 1 for tests
        
        // Setup Neo4j mocks
        when(neo4jDriver.session()).thenReturn(neo4jSession);
    }

    @AfterEach
    void tearDown() {
        processedCDRs.clear();
        fraudAlerts.clear();
    }

    @Test
    @DisplayName("Test CDR Stream Processing Pipeline")
    void testCDRStreamProcessing() throws Exception {
        // Create test CDR data
        List<CDRData> testCDRs = Arrays.asList(
            new CDRData("call_1", "1234567890", "0987654321", 300, 5.50, System.currentTimeMillis()),
            new CDRData("call_2", "1111111111", "2222222222", 600, 12.00, System.currentTimeMillis()),
            new CDRData("call_3", "3333333333", "4444444444", 60, 1.50, System.currentTimeMillis())
        );

        // Create test source
        SourceFunction<CDRData> testSource = new TestCDRSource(testCDRs);

        // Create test sink
        SinkFunction<CDRData> testSink = new TestCDRSink(processedCDRs);

        // Build processing pipeline
        env.addSource(testSource)
           .map(new CDREnrichmentFunction())
           .filter(new CDRValidationFunction())
           .addSink(testSink);

        // Execute the job
        env.execute("Test CDR Processing");

        // Verify results
        assertEquals(3, processedCDRs.size());
        
        CDRData processedCDR = processedCDRs.poll();
        assertNotNull(processedCDR);
        assertTrue(processedCDR.getCallId().startsWith("call_"));
        assertTrue(processedCDR.getDuration() > 0);
        assertTrue(processedCDR.getCost() > 0);
    }

    @Test
    @DisplayName("Test Fraud Detection Processing")
    void testFraudDetectionProcessing() throws Exception {
        // Create suspicious CDR data
        List<CDRData> suspiciousCDRs = Arrays.asList(
            new CDRData("fraud_1", "1234567890", "0987654321", 3600, 150.00, System.currentTimeMillis()), // Long expensive call
            new CDRData("fraud_2", "1234567890", "1111111111", 3600, 160.00, System.currentTimeMillis()), // Another long call from same number
            new CDRData("normal_1", "5555555555", "6666666666", 60, 2.00, System.currentTimeMillis())     // Normal call
        );

        // Create test source and sink
        SourceFunction<CDRData> testSource = new TestCDRSource(suspiciousCDRs);
        SinkFunction<FraudAlert> alertSink = new TestAlertSink(fraudAlerts);

        // Build fraud detection pipeline
        env.addSource(testSource)
           .keyBy(CDRData::getCallerNumber)
           .window(TumblingEventTimeWindows.of(Time.minutes(5)))
           .process(new FraudDetectionProcessFunction())
           .filter(alert -> alert.getFraudScore() > 0.7) // Only high-risk alerts
           .addSink(alertSink);

        // Execute the job
        env.execute("Test Fraud Detection");

        // Verify fraud alerts were generated
        assertTrue(fraudAlerts.size() > 0);
        
        FraudAlert alert = fraudAlerts.poll();
        assertNotNull(alert);
        assertTrue(alert.getFraudScore() > 0.7);
        assertEquals("1234567890", alert.getCallerNumber());
    }

    @Test
    @DisplayName("Test Windowing Operations")
    void testWindowingOperations() throws Exception {
        // Create CDR data with specific timestamps for windowing
        long baseTime = System.currentTimeMillis();
        List<CDRData> timedCDRs = Arrays.asList(
            new CDRData("w1_call1", "1111111111", "2222222222", 120, 3.00, baseTime),
            new CDRData("w1_call2", "1111111111", "3333333333", 180, 4.50, baseTime + 30000), // 30 seconds later
            new CDRData("w2_call1", "1111111111", "4444444444", 240, 6.00, baseTime + 300000) // 5 minutes later
        );

        // Test tumbling window aggregation
        SourceFunction<CDRData> testSource = new TestCDRSource(timedCDRs);
        SinkFunction<CDRAggregation> aggregationSink = new TestAggregationSink();

        env.addSource(testSource)
           .assignTimestampsAndWatermarks(new CDRTimestampExtractor())
           .keyBy(CDRData::getCallerNumber)
           .window(TumblingEventTimeWindows.of(Time.minutes(5)))
           .aggregate(new CDRAggregationFunction())
           .addSink(aggregationSink);

        env.execute("Test Windowing");

        // Verify windowing behavior
        // This would require collecting results and verifying window boundaries
    }

    @Test
    @DisplayName("Test Neo4j Sink Functionality")
    void testNeo4jSink() throws Exception {
        // Create test relationship data
        List<CallRelationship> relationships = Arrays.asList(
            new CallRelationship("1111111111", "2222222222", 1, 300, 5.50),
            new CallRelationship("2222222222", "3333333333", 2, 450, 8.25)
        );

        // Create test source
        SourceFunction<CallRelationship> testSource = new TestRelationshipSource(relationships);

        // Create Neo4j sink with mocked driver
        Neo4jSink<CallRelationship> neo4jSink = new Neo4jSink<>(neo4jDriver, new CallRelationshipCypherQuery());

        env.addSource(testSource)
           .addSink(neo4jSink);

        env.execute("Test Neo4j Sink");

        // Verify Neo4j interactions
        verify(neo4jDriver, atLeastOnce()).session();
        verify(neo4jSession, atLeastOnce()).writeTransaction(any());
    }

    @Test
    @DisplayName("Test Kafka Sink Functionality")
    void testKafkaSink() throws Exception {
        // Create test alert data
        List<FraudAlert> alerts = Arrays.asList(
            new FraudAlert("alert_1", "1111111111", 0.85, "high", "Suspicious pattern", System.currentTimeMillis()),
            new FraudAlert("alert_2", "2222222222", 0.92, "critical", "Fraud detected", System.currentTimeMillis())
        );

        // Create test source
        SourceFunction<FraudAlert> testSource = new TestAlertSource(alerts);

        // Create Kafka properties
        Properties kafkaProps = new Properties();
        kafkaProps.setProperty("bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS);
        kafkaProps.setProperty("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        kafkaProps.setProperty("value.serializer", "com.fraudguard.serialization.FraudAlertSerializer");

        // Create Kafka sink
        FlinkKafkaProducer<FraudAlert> kafkaSink = new FlinkKafkaProducer<>(
            "fraud-alerts",
            new FraudAlertSerializationSchema(),
            kafkaProps,
            FlinkKafkaProducer.Semantic.EXACTLY_ONCE
        );

        env.addSource(testSource)
           .addSink(kafkaSink);

        env.execute("Test Kafka Sink");

        // Verify Kafka producer interactions (would require Kafka test container)
    }

    @Test
    @DisplayName("Test CDR Validation Function")
    void testCDRValidation() {
        CDRValidationFunction validator = new CDRValidationFunction();

        // Valid CDR
        CDRData validCDR = new CDRData("call_1", "1234567890", "0987654321", 300, 5.50, System.currentTimeMillis());
        assertTrue(validator.filter(validCDR));

        // Invalid CDR - negative duration
        CDRData invalidCDR1 = new CDRData("call_2", "1234567890", "0987654321", -100, 5.50, System.currentTimeMillis());
        assertFalse(validator.filter(invalidCDR1));

        // Invalid CDR - negative cost
        CDRData invalidCDR2 = new CDRData("call_3", "1234567890", "0987654321", 300, -2.50, System.currentTimeMillis());
        assertFalse(validator.filter(invalidCDR2));

        // Invalid CDR - empty phone numbers
        CDRData invalidCDR3 = new CDRData("call_4", "", "0987654321", 300, 5.50, System.currentTimeMillis());
        assertFalse(validator.filter(invalidCDR3));
    }

    @Test
    @DisplayName("Test CDR Enrichment Function")
    void testCDREnrichment() throws Exception {
        CDREnrichmentFunction enricher = new CDREnrichmentFunction();
        
        CDRData originalCDR = new CDRData("call_1", "1234567890", "0987654321", 300, 5.50, System.currentTimeMillis());
        CDRData enrichedCDR = enricher.map(originalCDR);

        assertNotNull(enrichedCDR);
        assertEquals(originalCDR.getCallId(), enrichedCDR.getCallId());
        
        // Verify enrichment added metadata
        assertNotNull(enrichedCDR.getGeolocation());
        assertNotNull(enrichedCDR.getCallType());
        assertTrue(enrichedCDR.getRiskScore() >= 0.0 && enrichedCDR.getRiskScore() <= 1.0);
    }

    @Test
    @DisplayName("Test Fraud Pattern Detection")
    void testFraudPatternDetection() {
        FraudPatternDetector detector = new FraudPatternDetector();

        // Test high-frequency calling pattern
        List<CDRData> highFrequencyCalls = new ArrayList<>();
        long baseTime = System.currentTimeMillis();
        for (int i = 0; i < 50; i++) {
            highFrequencyCalls.add(new CDRData("call_" + i, "1111111111", "target_" + i, 60, 2.0, baseTime + i * 1000));
        }

        FraudPattern pattern = detector.detectPattern(highFrequencyCalls);
        assertNotNull(pattern);
        assertEquals("HIGH_FREQUENCY_CALLING", pattern.getPatternType());
        assertTrue(pattern.getRiskScore() > 0.8);

        // Test normal calling pattern
        List<CDRData> normalCalls = Arrays.asList(
            new CDRData("normal_1", "2222222222", "3333333333", 180, 4.50, baseTime),
            new CDRData("normal_2", "2222222222", "4444444444", 240, 6.00, baseTime + 3600000) // 1 hour later
        );

        FraudPattern normalPattern = detector.detectPattern(normalCalls);
        assertNull(normalPattern); // No fraud pattern detected
    }

    @Test
    @DisplayName("Test Streaming Job Performance")
    void testStreamingPerformance() throws Exception {
        // Create large dataset for performance testing
        int numberOfRecords = 10000;
        List<CDRData> largeCDRDataset = new ArrayList<>();
        
        for (int i = 0; i < numberOfRecords; i++) {
            largeCDRDataset.add(new CDRData(
                "perf_call_" + i,
                "caller_" + (i % 1000),
                "callee_" + ((i + 500) % 1000),
                60 + (i % 300),
                1.0 + (i % 10),
                System.currentTimeMillis() + i * 1000
            ));
        }

        long startTime = System.currentTimeMillis();

        SourceFunction<CDRData> testSource = new TestCDRSource(largeCDRDataset);
        SinkFunction<CDRData> testSink = new TestCDRSink(processedCDRs);

        env.addSource(testSource)
           .map(new CDREnrichmentFunction())
           .filter(new CDRValidationFunction())
           .addSink(testSink);

        env.execute("Performance Test");

        long endTime = System.currentTimeMillis();
        long processingTime = endTime - startTime;

        // Verify performance criteria
        assertEquals(numberOfRecords, processedCDRs.size());
        assertTrue(processingTime < 30000); // Should complete within 30 seconds
        
        double throughput = (double) numberOfRecords / (processingTime / 1000.0);
        assertTrue(throughput > 100); // Should process more than 100 records per second
    }

    @Test
    @DisplayName("Test Error Handling and Recovery")
    void testErrorHandling() throws Exception {
        // Create CDR data with some invalid records
        List<CDRData> mixedCDRData = Arrays.asList(
            new CDRData("valid_1", "1111111111", "2222222222", 300, 5.50, System.currentTimeMillis()),
            new CDRData("invalid_1", null, "2222222222", 300, 5.50, System.currentTimeMillis()), // null caller
            new CDRData("valid_2", "3333333333", "4444444444", 180, 3.25, System.currentTimeMillis()),
            new CDRData("invalid_2", "5555555555", "6666666666", -100, 2.50, System.currentTimeMillis()) // negative duration
        );

        SourceFunction<CDRData> testSource = new TestCDRSource(mixedCDRData);
        SinkFunction<CDRData> validSink = new TestCDRSink(processedCDRs);
        SinkFunction<CDRData> errorSink = new TestCDRSink(new LinkedBlockingQueue<>());

        // Create processing pipeline with error handling
        DataStream<CDRData> cdrStream = env.addSource(testSource);
        
        // Split stream into valid and invalid records
        OutputTag<CDRData> errorTag = new OutputTag<CDRData>("errors"){};
        
        SingleOutputStreamOperator<CDRData> processedStream = cdrStream
            .process(new CDRProcessFunctionWithErrorHandling(errorTag));

        processedStream.addSink(validSink);
        processedStream.getSideOutput(errorTag).addSink(errorSink);

        env.execute("Error Handling Test");

        // Verify only valid records were processed
        assertEquals(2, processedCDRs.size()); // Only 2 valid records
    }

    // Test helper classes
    private static class TestCDRSource implements SourceFunction<CDRData> {
        private final List<CDRData> testData;
        private volatile boolean running = true;

        public TestCDRSource(List<CDRData> testData) {
            this.testData = testData;
        }

        @Override
        public void run(SourceContext<CDRData> ctx) throws Exception {
            for (CDRData cdr : testData) {
                if (!running) break;
                ctx.collect(cdr);
            }
        }

        @Override
        public void cancel() {
            running = false;
        }
    }

    private static class TestCDRSink implements SinkFunction<CDRData> {
        private final BlockingQueue<CDRData> output;

        public TestCDRSink(BlockingQueue<CDRData> output) {
            this.output = output;
        }

        @Override
        public void invoke(CDRData value, Context context) throws Exception {
            output.offer(value);
        }
    }

    private static class TestAlertSink implements SinkFunction<FraudAlert> {
        private final BlockingQueue<FraudAlert> output;

        public TestAlertSink(BlockingQueue<FraudAlert> output) {
            this.output = output;
        }

        @Override
        public void invoke(FraudAlert value, Context context) throws Exception {
            output.offer(value);
        }
    }
}