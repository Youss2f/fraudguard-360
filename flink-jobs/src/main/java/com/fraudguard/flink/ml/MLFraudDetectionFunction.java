package com.fraudguard.flink.ml;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.async.AsyncFunction;
import org.apache.flink.streaming.api.functions.async.ResultFuture;
import org.apache.flink.streaming.api.functions.async.RichAsyncFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Collections;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Flink connector for ML fraud detection service
 * Processes CDR records and enriches them with fraud predictions
 */
public class MLFraudDetectionFunction extends RichAsyncFunction<JsonNode, JsonNode> {
    
    private static final Logger LOG = LoggerFactory.getLogger(MLFraudDetectionFunction.class);
    
    private final String mlServiceUrl;
    private final int timeout;
    private final int maxRetries;
    
    private transient HttpClient httpClient;
    private transient ObjectMapper objectMapper;
    private transient ExecutorService executorService;
    
    // Performance metrics
    private long totalRequests = 0;
    private long successfulRequests = 0;
    private long failedRequests = 0;
    private long totalLatency = 0;
    
    public MLFraudDetectionFunction(String mlServiceUrl, int timeout, int maxRetries) {
        this.mlServiceUrl = mlServiceUrl;
        this.timeout = timeout;
        this.maxRetries = maxRetries;
    }
    
    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        
        // Initialize HTTP client
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(timeout))
            .build();
        
        this.objectMapper = new ObjectMapper();
        this.executorService = Executors.newFixedThreadPool(10);
        
        LOG.info("ML Fraud Detection Function initialized with URL: {}", mlServiceUrl);
    }
    
    @Override
    public void close() throws Exception {
        super.close();
        
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
        
        // Log performance metrics
        LOG.info("ML Function Performance Metrics:");
        LOG.info("  Total Requests: {}", totalRequests);
        LOG.info("  Successful Requests: {}", successfulRequests);
        LOG.info("  Failed Requests: {}", failedRequests);
        LOG.info("  Success Rate: {:.2f}%", 
                 totalRequests > 0 ? (successfulRequests * 100.0 / totalRequests) : 0.0);
        LOG.info("  Average Latency: {:.2f}ms", 
                 successfulRequests > 0 ? (totalLatency / (double) successfulRequests) : 0.0);
    }
    
    @Override
    public void asyncInvoke(JsonNode input, ResultFuture<JsonNode> resultFuture) throws Exception {
        long startTime = System.currentTimeMillis();
        totalRequests++;
        
        // Extract CDR data for ML prediction
        ObjectNode mlRequest = createMLRequest(input);
        
        CompletableFuture
            .supplyAsync(() -> callMLService(mlRequest), executorService)
            .thenAccept(prediction -> {
                try {
                    if (prediction != null) {
                        // Enrich original record with prediction
                        ObjectNode enrichedRecord = enrichRecord(input, prediction);
                        
                        successfulRequests++;
                        totalLatency += (System.currentTimeMillis() - startTime);
                        
                        resultFuture.complete(Collections.singleton(enrichedRecord));
                        
                        // Log high-risk predictions
                        if (prediction.has("alert_level") && 
                            ("HIGH".equals(prediction.get("alert_level").asText()) ||
                             "CRITICAL".equals(prediction.get("alert_level").asText()))) {
                            LOG.warn("HIGH RISK FRAUD DETECTED: User {} - Risk Score: {:.1f}% - Type: {}",
                                    prediction.get("user_id").asText(),
                                    prediction.get("risk_score").asDouble(),
                                    prediction.get("fraud_type").asText());
                        }
                        
                    } else {
                        failedRequests++;
                        // Return original record on prediction failure
                        resultFuture.complete(Collections.singleton(input));
                    }
                } catch (Exception e) {
                    LOG.error("Error processing ML prediction result", e);
                    failedRequests++;
                    resultFuture.complete(Collections.singleton(input));
                }
            })
            .exceptionally(throwable -> {
                LOG.error("ML service call failed", throwable);
                failedRequests++;
                // Return original record on failure
                resultFuture.complete(Collections.singleton(input));
                return null;
            });
    }
    
    private ObjectNode createMLRequest(JsonNode cdrRecord) {
        ObjectNode mlRequest = objectMapper.createObjectNode();
        
        try {
            // Extract required fields for ML prediction
            mlRequest.put("user_id", cdrRecord.get("user_id").asText());
            mlRequest.put("call_duration", cdrRecord.get("call_duration").asDouble());
            mlRequest.put("call_cost", cdrRecord.get("call_cost").asDouble());
            
            // Handle potentially missing fields with defaults
            mlRequest.put("calls_per_day", getIntValueOrDefault(cdrRecord, "calls_per_day", 1));
            mlRequest.put("unique_numbers_called", getIntValueOrDefault(cdrRecord, "unique_numbers_called", 1));
            mlRequest.put("international_calls", getIntValueOrDefault(cdrRecord, "international_calls", 0));
            mlRequest.put("night_calls", getIntValueOrDefault(cdrRecord, "night_calls", 0));
            mlRequest.put("weekend_calls", getIntValueOrDefault(cdrRecord, "weekend_calls", 0));
            mlRequest.put("call_frequency_variance", getDoubleValueOrDefault(cdrRecord, "call_frequency_variance", 1.0));
            mlRequest.put("location_changes", getIntValueOrDefault(cdrRecord, "location_changes", 0));
            mlRequest.put("avg_call_gap", getDoubleValueOrDefault(cdrRecord, "avg_call_gap", 60.0));
            mlRequest.put("network_connections", getIntValueOrDefault(cdrRecord, "network_connections", 1));
            mlRequest.put("suspicious_patterns", getIntValueOrDefault(cdrRecord, "suspicious_patterns", 0));
            
            // Add timestamp
            mlRequest.put("timestamp", cdrRecord.get("timestamp").asText());
            
        } catch (Exception e) {
            LOG.error("Error creating ML request from CDR record", e);
        }
        
        return mlRequest;
    }
    
    private int getIntValueOrDefault(JsonNode node, String fieldName, int defaultValue) {
        JsonNode field = node.get(fieldName);
        return (field != null && !field.isNull()) ? field.asInt() : defaultValue;
    }
    
    private double getDoubleValueOrDefault(JsonNode node, String fieldName, double defaultValue) {
        JsonNode field = node.get(fieldName);
        return (field != null && !field.isNull()) ? field.asDouble() : defaultValue;
    }
    
    private JsonNode callMLService(ObjectNode mlRequest) {
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                String requestBody = objectMapper.writeValueAsString(mlRequest);
                
                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(mlServiceUrl + "/predict"))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(timeout))
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .build();
                
                HttpResponse<String> response = httpClient.send(request, 
                    HttpResponse.BodyHandlers.ofString());
                
                if (response.statusCode() == 200) {
                    return objectMapper.readTree(response.body());
                } else {
                    LOG.warn("ML service returned status code: {} for user: {}", 
                            response.statusCode(), mlRequest.get("user_id").asText());
                    
                    if (attempt < maxRetries) {
                        Thread.sleep(1000 * (attempt + 1)); // Exponential backoff
                        continue;
                    }
                }
                
            } catch (Exception e) {
                LOG.error("Attempt {} failed for ML service call", attempt + 1, e);
                
                if (attempt < maxRetries) {
                    try {
                        Thread.sleep(1000 * (attempt + 1)); // Exponential backoff
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                } else {
                    LOG.error("All {} attempts failed for ML service call", maxRetries + 1);
                }
            }
        }
        
        return null; // Return null on failure
    }
    
    private ObjectNode enrichRecord(JsonNode originalRecord, JsonNode prediction) {
        ObjectNode enrichedRecord = (ObjectNode) originalRecord.deepCopy();
        
        // Add fraud prediction fields
        ObjectNode fraudPrediction = objectMapper.createObjectNode();
        fraudPrediction.put("fraud_probability", prediction.get("fraud_probability").asDouble());
        fraudPrediction.put("risk_score", prediction.get("risk_score").asDouble());
        fraudPrediction.put("fraud_type", prediction.get("fraud_type").asText());
        fraudPrediction.put("confidence", prediction.get("confidence").asDouble());
        fraudPrediction.put("alert_level", prediction.get("alert_level").asText());
        fraudPrediction.put("recommendation", prediction.get("recommendation").asText());
        
        // Add risk factors
        if (prediction.has("risk_factors")) {
            fraudPrediction.set("risk_factors", prediction.get("risk_factors"));
        }
        
        // Add model predictions
        if (prediction.has("model_predictions")) {
            fraudPrediction.set("model_predictions", prediction.get("model_predictions"));
        }
        
        // Add prediction timestamp
        fraudPrediction.put("prediction_timestamp", prediction.get("timestamp").asText());
        
        // Add to original record
        enrichedRecord.set("fraud_prediction", fraudPrediction);
        
        // Add processing metadata
        enrichedRecord.put("ml_processed", true);
        enrichedRecord.put("ml_processing_timestamp", System.currentTimeMillis());
        
        return enrichedRecord;
    }
    
    /**
     * Synchronous version for batch processing
     */
    public static class SyncMLFraudDetectionFunction extends RichMapFunction<JsonNode, JsonNode> {
        
        private final MLFraudDetectionFunction asyncFunction;
        private transient long processedRecords = 0;
        private transient long highRiskRecords = 0;
        
        public SyncMLFraudDetectionFunction(String mlServiceUrl, int timeout, int maxRetries) {
            this.asyncFunction = new MLFraudDetectionFunction(mlServiceUrl, timeout, maxRetries);
        }
        
        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            asyncFunction.open(parameters);
        }
        
        @Override
        public void close() throws Exception {
            super.close();
            asyncFunction.close();
            
            LOG.info("Sync ML Function Statistics:");
            LOG.info("  Total Records Processed: {}", processedRecords);
            LOG.info("  High Risk Records: {}", highRiskRecords);
            LOG.info("  High Risk Rate: {:.2f}%", 
                     processedRecords > 0 ? (highRiskRecords * 100.0 / processedRecords) : 0.0);
        }
        
        @Override
        public JsonNode map(JsonNode value) throws Exception {
            processedRecords++;
            
            // Create ML request
            ObjectNode mlRequest = asyncFunction.createMLRequest(value);
            
            // Call ML service synchronously
            JsonNode prediction = asyncFunction.callMLService(mlRequest);
            
            if (prediction != null) {
                ObjectNode enrichedRecord = asyncFunction.enrichRecord(value, prediction);
                
                // Check if high risk
                if (prediction.has("alert_level")) {
                    String alertLevel = prediction.get("alert_level").asText();
                    if ("HIGH".equals(alertLevel) || "CRITICAL".equals(alertLevel)) {
                        highRiskRecords++;
                    }
                }
                
                return enrichedRecord;
            } else {
                // Return original record on failure
                return value;
            }
        }
    }
}