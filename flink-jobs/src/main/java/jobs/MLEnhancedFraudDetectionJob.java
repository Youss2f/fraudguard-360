package jobs;

import com.fraudguard.flink.ml.MLFraudDetectionFunction;
import models.CDR;
import models.FraudAlert;
import models.FraudStatistics;
import processors.AdvancedFraudDetectionProcessor;
import aggregators.FraudStatisticsAggregator;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.streaming.api.datastream.AsyncDataStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

/**
 * ML-Enhanced Fraud Detection Job
 * Integrates machine learning predictions with real-time stream processing
 */
public class MLEnhancedFraudDetectionJob {
    
    private static final Logger LOG = LoggerFactory.getLogger(MLEnhancedFraudDetectionJob.class);
    
    // Configuration constants
    private static final String KAFKA_BROKERS = "localhost:9092";
    private static final String INPUT_TOPIC = "cdr-events";
    private static final String OUTPUT_TOPIC = "fraud-alerts";
    private static final String ML_ALERTS_TOPIC = "ml-fraud-alerts";
    private static final String ML_SERVICE_URL = "http://localhost:8003";
    
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    public static void main(String[] args) throws Exception {
        // Set up the streaming execution environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // Configure for production
        env.setParallelism(4);
        env.enableCheckpointing(30000); // Checkpoint every 30 seconds
        
        LOG.info("Starting ML-Enhanced Fraud Detection Job");
        
        try {
            // Configure Kafka source
            KafkaSource<String> source = KafkaSource.<String>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setTopics(INPUT_TOPIC)
                .setGroupId("fraud-detection-ml-group")
                .setStartingOffsets(OffsetsInitializer.latest())
                .setValueOnlyDeserializer(new SimpleStringSchema())
                .build();
            
            // Read CDR data from Kafka
            DataStream<String> cdrStringStream = env
                .fromSource(source, WatermarkStrategy.noWatermarks(), "Kafka CDR Source");
            
            // Parse JSON CDR records
            DataStream<JsonNode> cdrJsonStream = cdrStringStream
                .map(new JsonParseFunction())
                .filter(jsonNode -> jsonNode != null)
                .name("Parse CDR JSON");
            
            // Enrich with ML fraud predictions (async)
            DataStream<JsonNode> mlEnrichedStream = AsyncDataStream.unorderedWait(
                cdrJsonStream,
                new MLFraudDetectionFunction(ML_SERVICE_URL, 5, 2), // 5s timeout, 2 retries
                10000, // 10s async timeout
                TimeUnit.MILLISECONDS,
                100 // max async requests
            ).name("ML Fraud Prediction");
            
            // Filter high-risk predictions for immediate alerting
            DataStream<JsonNode> highRiskStream = mlEnrichedStream
                .filter(new HighRiskFilterFunction())
                .name("High Risk Filter");
            
            // Convert to fraud alerts
            DataStream<JsonNode> fraudAlertStream = highRiskStream
                .map(new FraudAlertMapFunction())
                .name("Create Fraud Alerts");
            
            // Send high-risk alerts to dedicated topic
            KafkaSink<JsonNode> mlAlertSink = KafkaSink.<JsonNode>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                    .setTopic(ML_ALERTS_TOPIC)
                    .setValueSerializationSchema(new JsonSerializationSchema())
                    .build())
                .build();
            
            fraudAlertStream.sinkTo(mlAlertSink).name("ML Fraud Alerts Sink");
            
            // Send all enriched records to output topic
            KafkaSink<JsonNode> enrichedSink = KafkaSink.<JsonNode>builder()
                .setBootstrapServers(KAFKA_BROKERS)
                .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                    .setTopic(OUTPUT_TOPIC)
                    .setValueSerializationSchema(new JsonSerializationSchema())
                    .build())
                .build();
            
            mlEnrichedStream.sinkTo(enrichedSink).name("Enriched CDR Sink");
            
            // Real-time statistics aggregation
            DataStream<FraudStatistics> statisticsStream = mlEnrichedStream
                .map(new StatisticsMapFunction())
                .filter(stat -> stat != null)
                .windowAll(TumblingProcessingTimeWindows.of(Time.minutes(1)))
                .aggregate(new FraudStatisticsAggregator())
                .name("Fraud Statistics");
            
            // Print statistics for monitoring
            statisticsStream.print("Fraud Statistics");
            
            // Execute the job
            env.execute("ML-Enhanced Fraud Detection Job");
            
        } catch (Exception e) {
            LOG.error("Error in ML-Enhanced Fraud Detection Job", e);
            throw e;
        }
    }
    
    /**
     * Function to parse JSON strings into JsonNode objects
     */
    public static class JsonParseFunction implements MapFunction<String, JsonNode> {
        @Override
        public JsonNode map(String value) throws Exception {
            try {
                return objectMapper.readTree(value);
            } catch (Exception e) {
                LOG.warn("Failed to parse JSON: {}", value, e);
                return null;
            }
        }
    }
    
    /**
     * Filter function to identify high-risk predictions
     */
    public static class HighRiskFilterFunction implements FilterFunction<JsonNode> {
        @Override
        public boolean filter(JsonNode value) throws Exception {
            if (!value.has("fraud_prediction")) {
                return false;
            }
            
            JsonNode prediction = value.get("fraud_prediction");
            if (!prediction.has("alert_level")) {
                return false;
            }
            
            String alertLevel = prediction.get("alert_level").asText();
            return "HIGH".equals(alertLevel) || "CRITICAL".equals(alertLevel);
        }
    }
    
    /**
     * Map function to create fraud alert records
     */
    public static class FraudAlertMapFunction implements MapFunction<JsonNode, JsonNode> {
        @Override
        public JsonNode map(JsonNode value) throws Exception {
            ObjectNode alert = objectMapper.createObjectNode();
            
            // Extract basic information
            alert.put("alert_id", "fraud_" + System.currentTimeMillis() + "_" + 
                     value.get("user_id").asText());
            alert.put("user_id", value.get("user_id").asText());
            alert.put("timestamp", System.currentTimeMillis());
            alert.put("alert_type", "ML_FRAUD_DETECTION");
            
            // Extract fraud prediction details
            JsonNode prediction = value.get("fraud_prediction");
            alert.put("fraud_probability", prediction.get("fraud_probability").asDouble());
            alert.put("risk_score", prediction.get("risk_score").asDouble());
            alert.put("fraud_type", prediction.get("fraud_type").asText());
            alert.put("alert_level", prediction.get("alert_level").asText());
            alert.put("confidence", prediction.get("confidence").asDouble());
            alert.put("recommendation", prediction.get("recommendation").asText());
            
            // Add risk factors
            if (prediction.has("risk_factors")) {
                alert.set("risk_factors", prediction.get("risk_factors"));
            }
            
            // Add original CDR data for context
            ObjectNode cdrContext = objectMapper.createObjectNode();
            cdrContext.put("call_duration", value.get("call_duration").asDouble());
            cdrContext.put("call_cost", value.get("call_cost").asDouble());
            cdrContext.put("international_calls", value.get("international_calls").asInt());
            cdrContext.put("location_changes", value.get("location_changes").asInt());
            
            alert.set("cdr_context", cdrContext);
            
            return alert;
        }
    }
    
    /**
     * Map function to extract statistics from ML predictions
     */
    public static class StatisticsMapFunction implements MapFunction<JsonNode, FraudStatistics> {
        @Override
        public FraudStatistics map(JsonNode value) throws Exception {
            if (!value.has("fraud_prediction")) {
                return null;
            }
            
            JsonNode prediction = value.get("fraud_prediction");
            
            FraudStatistics stats = new FraudStatistics();
            stats.setTimestamp(System.currentTimeMillis());
            stats.setTotalCalls(1);
            stats.setTotalUsers(1);
            
            // Check if this is a fraud case
            String alertLevel = prediction.get("alert_level").asText();
            if ("HIGH".equals(alertLevel) || "CRITICAL".equals(alertLevel)) {
                stats.setFraudCalls(1);
                stats.setFraudUsers(1);
            }
            
            // Set risk metrics
            stats.setAverageRiskScore(prediction.get("risk_score").asDouble());
            stats.setAverageFraudProbability(prediction.get("fraud_probability").asDouble());
            
            return stats;
        }
    }
    
    /**
     * Serialization schema for JSON output to Kafka
     */
    public static class JsonSerializationSchema implements org.apache.flink.api.common.serialization.SerializationSchema<JsonNode> {
        @Override
        public byte[] serialize(JsonNode element) {
            try {
                return objectMapper.writeValueAsBytes(element);
            } catch (Exception e) {
                LOG.error("Failed to serialize JsonNode", e);
                return new byte[0];
            }
        }
    }
}