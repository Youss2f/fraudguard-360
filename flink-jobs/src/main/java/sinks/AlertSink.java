package sinks;

import models.FraudAlert;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Properties;

/**
 * Kafka sink for fraud alerts
 */
public class AlertSink implements SinkFunction<FraudAlert> {
    
    private static final Logger logger = LoggerFactory.getLogger(AlertSink.class);
    
    private final String bootstrapServers;
    private final String topic;
    private transient KafkaProducer<String, String> producer;
    private transient ObjectMapper objectMapper;

    public AlertSink(String bootstrapServers, String topic) {
        this.bootstrapServers = bootstrapServers;
        this.topic = topic;
    }

    @Override
    public void open(org.apache.flink.configuration.Configuration parameters) throws Exception {
        super.open(parameters);
        
        Properties props = new Properties();
        props.setProperty("bootstrap.servers", bootstrapServers);
        props.setProperty("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.setProperty("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.setProperty("acks", "all");
        props.setProperty("retries", "3");
        
        producer = new KafkaProducer<>(props);
        objectMapper = new ObjectMapper();
        
        logger.info("AlertSink initialized with topic: {}", topic);
    }

    @Override
    public void invoke(FraudAlert alert, Context context) throws Exception {
        try {
            String alertJson = objectMapper.writeValueAsString(alert);
            ProducerRecord<String, String> record = new ProducerRecord<>(topic, alert.getUserId(), alertJson);
            
            producer.send(record, (metadata, exception) -> {
                if (exception != null) {
                    logger.error("Failed to send alert to Kafka: {}", alert.getAlertId(), exception);
                } else {
                    logger.debug("Alert sent to Kafka successfully: {} to partition: {}", 
                               alert.getAlertId(), metadata.partition());
                }
            });
            
        } catch (Exception e) {
            logger.error("Error processing fraud alert: {}", alert.getAlertId(), e);
            throw e;
        }
    }

    @Override
    public void close() throws Exception {
        if (producer != null) {
            producer.flush();
            producer.close();
        }
        super.close();
    }
}