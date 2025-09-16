package com.fraudguard;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import com.fasterxml.jackson.databind.ObjectMapper;
import models.CDR;
import operators.FraudFeatureEnrichment;
import sinks.Neo4jSink;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fraudguard.models.GraphOperation;

// GraphOperation moved to its own public class in com.fraudguard.models

public class GraphProcessingJob {
    private static final Logger logger = LoggerFactory.getLogger(GraphProcessingJob.class);

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.enableCheckpointing(60000);  // Exactly-once

        // TODO: Add Kafka connector dependency and source; using placeholder stream for buildability
        DataStream<String> cdrJsonStream = env.fromElements(
            "{\"callerId\":\"A\",\"calleeId\":\"B\",\"duration\":30,\"timestamp\":\"2025-09-12T10:00:00Z\"}",
            "{\"callerId\":\"A\",\"calleeId\":\"C\",\"duration\":60,\"timestamp\":\"2025-09-12T10:01:00Z\"}"
        );

        DataStream<GraphOperation> graphStream = cdrJsonStream
            .keyBy((String json) -> {
                try {
                    CDR cdr = new ObjectMapper().readValue(json, CDR.class);
                    return cdr.getCallerId();
                } catch (Exception e) {
                    logger.error("Deserialization error", e);
                    return "error";
                }
            })
            .process(new FraudFeatureEnrichment());

        graphStream.addSink(new Neo4jSink("bolt://neo4j:7687", "neo4j", "password"));

        env.execute("FraudGuard Streaming Job");
    }
}
