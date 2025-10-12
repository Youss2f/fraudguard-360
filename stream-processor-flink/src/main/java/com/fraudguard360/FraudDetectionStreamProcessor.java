package com.fraudguard360;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.time.Time;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetInitializer;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * FraudGuard 360° - Real-time Fraud Detection Stream Processor
 * 
 * High-performance Flink application for real-time transaction fraud detection
 * Processes 5K-25K TPS with <200ms latency requirements
 * 
 * Features:
 * - Real-time transaction analysis
 * - Velocity fraud detection
 * - Pattern anomaly detection  
 * - ML model integration
 * - Alert generation
 * - Metrics collection
 */
public class FraudDetectionStreamProcessor {
    
    private static final Logger LOG = LoggerFactory.getLogger(FraudDetectionStreamProcessor.class);
    private static final ObjectMapper objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());
    
    // Configuration constants
    private static final String KAFKA_BROKERS = "localhost:9092";
    private static final String INPUT_TOPIC = "transactions";
    private static final String OUTPUT_TOPIC = "fraud-alerts";
    private static final String ENRICHED_TOPIC = "enriched-transactions";
    
    // Fraud detection thresholds
    private static final double HIGH_AMOUNT_THRESHOLD = 10000.0;
    private static final int VELOCITY_THRESHOLD_COUNT = 10; // transactions per minute
    private static final double VELOCITY_THRESHOLD_AMOUNT = 50000.0; // total amount per minute
    private static final long PATTERN_ANALYSIS_WINDOW_MINUTES = 15;
    
    // Output tags for side outputs
    private static final OutputTag<String> CRITICAL_ALERTS_TAG = new OutputTag<String>("critical-alerts") {};
    private static final OutputTag<String> SUSPICIOUS_PATTERNS_TAG = new OutputTag<String>("suspicious-patterns") {};
    private static final OutputTag<String> VELOCITY_VIOLATIONS_TAG = new OutputTag<String>("velocity-violations") {};

    public static void main(String[] args) throws Exception {
        
        LOG.info("Starting FraudGuard 360° Stream Processor...");
        
        // Set up the execution environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // Configure environment for high throughput
        env.setParallelism(4);
        env.enableCheckpointing(5000); // checkpoint every 5 seconds
        env.getCheckpointConfig().setMinPauseBetweenCheckpoints(1000);
        env.getCheckpointConfig().setCheckpointTimeout(60000);
        env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);
        
        // Configure Kafka source
        KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setTopics(INPUT_TOPIC)
                .setGroupId("fraud-detection-processor")
                .setStartingOffsets(OffsetInitializer.latest())
                .setValueOnlyDeserializer(new SimpleStringSchema())
                .build();

        // Create the transaction stream with watermarks
        DataStream<String> transactionStream = env
                .fromSource(kafkaSource, WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofSeconds(5)), "Kafka Source")
                .name("Transaction Source")
                .setParallelism(2);

        // Parse JSON transactions
        SingleOutputStreamOperator<Transaction> parsedTransactions = transactionStream
                .process(new TransactionParser())
                .name("Transaction Parser");

        // Enrich transactions with additional data
        SingleOutputStreamOperator<EnrichedTransaction> enrichedTransactions = parsedTransactions
                .process(new TransactionEnricher())
                .name("Transaction Enricher");

        // Key by user ID for stateful processing
        KeyedStream<EnrichedTransaction, String> keyedByUser = enrichedTransactions
                .keyBy(new UserKeySelector());

        // Velocity fraud detection
        SingleOutputStreamOperator<FraudAlert> velocityAlerts = keyedByUser
                .process(new VelocityFraudDetector())
                .name("Velocity Fraud Detector");

        // Amount-based fraud detection
        SingleOutputStreamOperator<FraudAlert> amountAlerts = enrichedTransactions
                .process(new AmountFraudDetector())
                .name("Amount Fraud Detector");

        // Pattern anomaly detection
        SingleOutputStreamOperator<FraudAlert> patternAlerts = keyedByUser
                .window(TumblingEventTimeWindows.of(Time.minutes(PATTERN_ANALYSIS_WINDOW_MINUTES)))
                .process(new PatternAnomalyDetector())
                .name("Pattern Anomaly Detector");

        // Union all fraud alerts
        DataStream<FraudAlert> allAlerts = velocityAlerts
                .union(amountAlerts)
                .union(patternAlerts);

        // ML-based fraud scoring
        SingleOutputStreamOperator<ScoredTransaction> scoredTransactions = enrichedTransactions
                .process(new MLFraudScorer())
                .name("ML Fraud Scorer");

        // Alert prioritization and routing
        SingleOutputStreamOperator<String> processedAlerts = allAlerts
                .process(new AlertProcessor())
                .name("Alert Processor");

        // Configure Kafka sinks
        KafkaSink<String> alertSink = KafkaSink.<String>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                        .setTopic(OUTPUT_TOPIC)
                        .setValueSerializationSchema(new SimpleStringSchema())
                        .build())
                .build();

        KafkaSink<String> enrichedSink = KafkaSink.<String>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                        .setTopic(ENRICHED_TOPIC)
                        .setValueSerializationSchema(new SimpleStringSchema())
                        .build())
                .build();

        // Output processed alerts
        processedAlerts.sinkTo(alertSink).name("Alert Sink");

        // Output enriched transactions
        scoredTransactions
                .map(tx -> objectMapper.writeValueAsString(tx))
                .sinkTo(enrichedSink)
                .name("Enriched Transaction Sink");

        // Side outputs for specialized processing
        processedAlerts.getSideOutput(CRITICAL_ALERTS_TAG)
                .print("CRITICAL ALERT");

        processedAlerts.getSideOutput(VELOCITY_VIOLATIONS_TAG)
                .print("VELOCITY VIOLATION");

        // Execute the pipeline
        LOG.info("Fraud Detection Stream Processor configured. Starting execution...");
        env.execute("FraudGuard 360° Stream Processor");
    }

    // Transaction parser
    public static class TransactionParser extends ProcessFunction<String, Transaction> {
        
        @Override
        public void processElement(String value, Context ctx, Collector<Transaction> out) throws Exception {
            try {
                JsonNode jsonNode = objectMapper.readTree(value);
                
                Transaction transaction = new Transaction();
                transaction.setTransactionId(jsonNode.get("transaction_id").asText());
                transaction.setUserId(jsonNode.get("user_id").asText());
                transaction.setMerchantId(jsonNode.get("merchant_id").asText());
                transaction.setAmount(jsonNode.get("amount").asDouble());
                transaction.setTimestamp(Instant.parse(jsonNode.get("timestamp").asText()));
                transaction.setMerchantCategory(jsonNode.get("merchant_category").asText());
                transaction.setDeviceFingerprint(jsonNode.get("device_fingerprint").asText());
                transaction.setIpAddress(jsonNode.get("ip_address").asText());
                
                if (jsonNode.has("location")) {
                    JsonNode location = jsonNode.get("location");
                    transaction.setLatitude(location.get("lat").asDouble());
                    transaction.setLongitude(location.get("lon").asDouble());
                }
                
                out.collect(transaction);
                
            } catch (Exception e) {
                LOG.error("Failed to parse transaction: {}", value, e);
                // In production, would send to dead letter queue
            }
        }
    }

    // Transaction enricher
    public static class TransactionEnricher extends ProcessFunction<Transaction, EnrichedTransaction> {
        
        private transient RiskScoreCalculator riskCalculator;
        
        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            this.riskCalculator = new RiskScoreCalculator();
        }
        
        @Override
        public void processElement(Transaction transaction, Context ctx, Collector<EnrichedTransaction> out) throws Exception {
            
            EnrichedTransaction enriched = new EnrichedTransaction(transaction);
            
            // Calculate base risk scores
            enriched.setAmountRisk(riskCalculator.calculateAmountRisk(transaction.getAmount()));
            enriched.setLocationRisk(riskCalculator.calculateLocationRisk(transaction.getLatitude(), transaction.getLongitude()));
            enriched.setMerchantRisk(riskCalculator.calculateMerchantRisk(transaction.getMerchantCategory()));
            enriched.setTimeRisk(riskCalculator.calculateTimeRisk(transaction.getTimestamp()));
            enriched.setDeviceRisk(riskCalculator.calculateDeviceRisk(transaction.getDeviceFingerprint()));
            
            // Set processing timestamp
            enriched.setProcessingTime(Instant.now());
            
            out.collect(enriched);
        }
    }

    // Velocity fraud detector
    public static class VelocityFraudDetector extends KeyedProcessFunction<String, EnrichedTransaction, FraudAlert> {
        
        private ValueState<VelocityState> velocityState;
        
        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            
            ValueStateDescriptor<VelocityState> descriptor = new ValueStateDescriptor<>(
                    "velocity-state",
                    VelocityState.class
            );
            velocityState = getRuntimeContext().getState(descriptor);
        }
        
        @Override
        public void processElement(EnrichedTransaction transaction, Context ctx, Collector<FraudAlert> out) throws Exception {
            
            VelocityState state = velocityState.value();
            if (state == null) {
                state = new VelocityState();
            }
            
            long currentTime = transaction.getTimestamp().toEpochMilli();
            long oneMinuteAgo = currentTime - TimeUnit.MINUTES.toMillis(1);
            
            // Clean old transactions
            state.getTransactions().removeIf(tx -> tx.getTimestamp() < oneMinuteAgo);
            
            // Add current transaction
            state.getTransactions().add(new VelocityTransaction(currentTime, transaction.getAmount()));
            
            // Check velocity thresholds
            int transactionCount = state.getTransactions().size();
            double totalAmount = state.getTransactions().stream()
                    .mapToDouble(VelocityTransaction::getAmount)
                    .sum();
            
            boolean velocityViolation = transactionCount > VELOCITY_THRESHOLD_COUNT || 
                                       totalAmount > VELOCITY_THRESHOLD_AMOUNT;
            
            if (velocityViolation) {
                FraudAlert alert = new FraudAlert();
                alert.setTransactionId(transaction.getTransactionId());
                alert.setUserId(transaction.getUserId());
                alert.setAlertType("VELOCITY_FRAUD");
                alert.setRiskScore(0.8 + (transactionCount > VELOCITY_THRESHOLD_COUNT ? 0.1 : 0.0) + 
                                  (totalAmount > VELOCITY_THRESHOLD_AMOUNT ? 0.1 : 0.0));
                alert.setReason(String.format("Velocity violation: %d transactions, $%.2f total in 1 minute", 
                               transactionCount, totalAmount));
                alert.setTimestamp(Instant.now());
                alert.setSeverity(alert.getRiskScore() > 0.9 ? "CRITICAL" : "HIGH");
                
                out.collect(alert);
                
                // Side output for velocity violations
                ctx.output(VELOCITY_VIOLATIONS_TAG, objectMapper.writeValueAsString(alert));
            }
            
            velocityState.update(state);
        }
    }

    // Amount-based fraud detector
    public static class AmountFraudDetector extends ProcessFunction<EnrichedTransaction, FraudAlert> {
        
        @Override
        public void processElement(EnrichedTransaction transaction, Context ctx, Collector<FraudAlert> out) throws Exception {
            
            if (transaction.getAmount() > HIGH_AMOUNT_THRESHOLD) {
                FraudAlert alert = new FraudAlert();
                alert.setTransactionId(transaction.getTransactionId());
                alert.setUserId(transaction.getUserId());
                alert.setAlertType("HIGH_AMOUNT");
                alert.setRiskScore(Math.min(0.5 + (transaction.getAmount() / HIGH_AMOUNT_THRESHOLD) * 0.4, 1.0));
                alert.setReason(String.format("High amount transaction: $%.2f", transaction.getAmount()));
                alert.setTimestamp(Instant.now());
                alert.setSeverity(transaction.getAmount() > HIGH_AMOUNT_THRESHOLD * 5 ? "CRITICAL" : "HIGH");
                
                out.collect(alert);
            }
        }
    }

    // Pattern anomaly detector
    public static class PatternAnomalyDetector extends ProcessWindowFunction<EnrichedTransaction, FraudAlert, String, TimeWindow> {
        
        @Override
        public void process(String userId, Context context, Iterable<EnrichedTransaction> elements, Collector<FraudAlert> out) throws Exception {
            
            List<EnrichedTransaction> transactions = new ArrayList<>();
            elements.forEach(transactions::add);
            
            if (transactions.size() < 3) {
                return; // Need minimum transactions for pattern analysis
            }
            
            // Analyze patterns
            PatternAnalyzer analyzer = new PatternAnalyzer();
            PatternAnalysisResult result = analyzer.analyzePatterns(transactions);
            
            if (result.isAnomalous()) {
                for (String anomaly : result.getAnomalies()) {
                    FraudAlert alert = new FraudAlert();
                    alert.setTransactionId("PATTERN_" + System.currentTimeMillis());
                    alert.setUserId(userId);
                    alert.setAlertType("PATTERN_ANOMALY");
                    alert.setRiskScore(result.getAnomalyScore());
                    alert.setReason("Pattern anomaly detected: " + anomaly);
                    alert.setTimestamp(Instant.now());
                    alert.setSeverity(result.getAnomalyScore() > 0.8 ? "HIGH" : "MEDIUM");
                    
                    out.collect(alert);
                    
                    // Side output for pattern analysis
                    context.output(SUSPICIOUS_PATTERNS_TAG, objectMapper.writeValueAsString(alert));
                }
            }
        }
    }

    // ML fraud scorer
    public static class MLFraudScorer extends ProcessFunction<EnrichedTransaction, ScoredTransaction> {
        
        private transient MLModelProxy mlModel;
        
        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            this.mlModel = new MLModelProxy();
        }
        
        @Override
        public void processElement(EnrichedTransaction transaction, Context ctx, Collector<ScoredTransaction> out) throws Exception {
            
            try {
                // Call ML service for fraud scoring
                double fraudScore = mlModel.predictFraudProbability(transaction);
                
                ScoredTransaction scored = new ScoredTransaction(transaction);
                scored.setFraudScore(fraudScore);
                scored.setModelVersion(mlModel.getModelVersion());
                scored.setScoringTime(Instant.now());
                
                out.collect(scored);
                
            } catch (Exception e) {
                LOG.error("ML scoring failed for transaction {}", transaction.getTransactionId(), e);
                
                // Fallback to rule-based scoring
                ScoredTransaction scored = new ScoredTransaction(transaction);
                scored.setFraudScore(calculateFallbackScore(transaction));
                scored.setModelVersion("FALLBACK_RULES");
                scored.setScoringTime(Instant.now());
                
                out.collect(scored);
            }
        }
        
        private double calculateFallbackScore(EnrichedTransaction transaction) {
            // Simple rule-based fallback scoring
            double score = 0.0;
            
            score += transaction.getAmountRisk() * 0.3;
            score += transaction.getLocationRisk() * 0.2;
            score += transaction.getMerchantRisk() * 0.2;
            score += transaction.getTimeRisk() * 0.1;
            score += transaction.getDeviceRisk() * 0.2;
            
            return Math.min(score, 1.0);
        }
    }

    // Alert processor
    public static class AlertProcessor extends ProcessFunction<FraudAlert, String> {
        
        @Override
        public void processElement(FraudAlert alert, Context ctx, Collector<String> out) throws Exception {
            
            // Enrich alert with additional context
            alert.setProcessingTime(Instant.now());
            alert.setProcessorId("flink-fraud-detector");
            
            // Route critical alerts to side output
            if ("CRITICAL".equals(alert.getSeverity())) {
                ctx.output(CRITICAL_ALERTS_TAG, objectMapper.writeValueAsString(alert));
            }
            
            // Convert to JSON and output
            String alertJson = objectMapper.writeValueAsString(alert);
            out.collect(alertJson);
            
            // Log alert for monitoring
            LOG.warn("Fraud Alert Generated: {} - {} - Risk: {:.2f}", 
                    alert.getAlertType(), alert.getTransactionId(), alert.getRiskScore());
        }
    }

    // Key selector for user-based keying
    public static class UserKeySelector implements KeySelector<EnrichedTransaction, String> {
        @Override
        public String getKey(EnrichedTransaction transaction) throws Exception {
            return transaction.getUserId();
        }
    }
}