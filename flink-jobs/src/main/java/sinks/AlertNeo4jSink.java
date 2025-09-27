package sinks;

import models.FraudAlert;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Values;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Neo4j sink for fraud alerts
 */
public class AlertNeo4jSink implements SinkFunction<FraudAlert> {
    private static final Logger logger = LoggerFactory.getLogger(AlertNeo4jSink.class);
    
    private final String uri;
    private final String username;
    private final String password;
    private transient Driver driver;

    public AlertNeo4jSink(String uri, String username, String password) {
        this.uri = uri;
        this.username = username;
        this.password = password;
    }

    @Override
    public void invoke(FraudAlert alert, Context context) throws Exception {
        if (driver == null) {
            driver = GraphDatabase.driver(uri, AuthTokens.basic(username, password));
        }
        
        try (var session = driver.session()) {
            // Create alert node and link to user
            session.run(
                "MATCH (user:User {id: $userId}) " +
                "CREATE (alert:FraudAlert {" +
                "  id: $alertId, " +
                "  fraudType: $fraudType, " +
                "  severity: $severity, " +
                "  confidenceScore: $confidenceScore, " +
                "  riskScore: $riskScore, " +
                "  description: $description, " +
                "  status: $status, " +
                "  timestamp: datetime($timestamp)" +
                "}) " +
                "CREATE (user)-[:HAS_ALERT]->(alert)",
                Values.parameters(
                    "userId", alert.getUserId(),
                    "alertId", alert.getAlertId(),
                    "fraudType", alert.getFraudType(),
                    "severity", alert.getSeverity(),
                    "confidenceScore", alert.getConfidenceScore(),
                    "riskScore", alert.getRiskScore(),
                    "description", alert.getDescription(),
                    "status", alert.getStatus(),
                    "timestamp", alert.getTimestamp()
                )
            );

            logger.debug("Fraud alert written to Neo4j: {}", alert.getAlertId());
            
        } catch (Exception e) {
            logger.error("Error writing alert to Neo4j: {}", alert.getAlertId(), e);
            throw e;
        }
    }
}