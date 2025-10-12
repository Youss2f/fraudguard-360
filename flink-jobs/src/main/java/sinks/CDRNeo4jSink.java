package sinks;

import models.CDR;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Values;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Neo4j sink for CDR data
 */
public class CDRNeo4jSink implements SinkFunction<CDR> {
    private static final Logger logger = LoggerFactory.getLogger(CDRNeo4jSink.class);
    
    private final String uri;
    private final String username;
    private final String password;
    private transient Driver driver;

    public CDRNeo4jSink(String uri, String username, String password) {
        this.uri = uri;
        this.username = username;
        this.password = password;
    }

    @Override
    public void open(org.apache.flink.configuration.Configuration parameters) throws Exception {
        driver = GraphDatabase.driver(uri, AuthTokens.basic(username, password));
        logger.info("Connected to Neo4j for CDR data at: {}", uri);
    }

    @Override
    public void invoke(CDR cdr, Context context) throws Exception {
        try (var session = driver.session()) {
            // Create nodes and relationships for CDR data
            session.run(
                "MERGE (caller:User {id: $callerId}) " +
                "MERGE (callee:User {id: $calleeId}) " +
                "CREATE (call:Call {" +
                "  id: $cdrId, " +
                "  type: $callType, " +
                "  startTime: $startTime, " +
                "  duration: $duration, " +
                "  cost: $cost, " +
                "  countryCode: $countryCode, " +
                "  timestamp: datetime($startTime)" +
                "}) " +
                "CREATE (caller)-[:MADE_CALL]->(call) " +
                "CREATE (call)-[:TO]->(callee)",
                Values.parameters(
                    "callerId", cdr.getCallerId(),
                    "calleeId", cdr.getCalleeId(),
                    "cdrId", cdr.getId(),
                    "callType", cdr.getCallType(),
                    "startTime", cdr.getStartTime(),
                    "duration", cdr.getDuration(),
                    "cost", cdr.getCost(),
                    "countryCode", cdr.getCountryCode()
                )
            );

            // Create device relationship if IMEI is present
            if (cdr.getDeviceImei() != null) {
                session.run(
                    "MATCH (user:User {id: $callerId}) " +
                    "MERGE (device:Device {imei: $imei}) " +
                    "MERGE (user)-[:USES_DEVICE]->(device)",
                    Values.parameters(
                        "callerId", cdr.getCallerId(),
                        "imei", cdr.getDeviceImei()
                    )
                );
            }

            // Create location relationship if present
            if (cdr.getLocationCaller() != null) {
                session.run(
                    "MATCH (user:User {id: $callerId}) " +
                    "MERGE (location:Location {name: $location}) " +
                    "MERGE (user)-[:LOCATED_AT {timestamp: datetime($startTime)}]->(location)",
                    Values.parameters(
                        "callerId", cdr.getCallerId(),
                        "location", cdr.getLocationCaller(),
                        "startTime", cdr.getStartTime()
                    )
                );
            }

            logger.debug("CDR data written to Neo4j: {}", cdr.getId());
            
        } catch (Exception e) {
            logger.error("Error writing CDR to Neo4j: {}", cdr.getId(), e);
            throw e;
        }
    }

    @Override
    public void close() throws Exception {
        if (driver != null) {
            driver.close();
        }
    }
}