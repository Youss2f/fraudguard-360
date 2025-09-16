package sinks;

import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Values;
import com.fraudguard.models.GraphOperation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Neo4jSink implements SinkFunction<GraphOperation> {
    private static final Logger logger = LoggerFactory.getLogger(Neo4jSink.class);
    private final transient Driver driver;

    public Neo4jSink(String uri, String user, String pass) {
        this.driver = GraphDatabase.driver(uri, AuthTokens.basic(user, pass));
    }

    @Override
    public void invoke(GraphOperation op, Context ctx) {
        try (var session = driver.session()) {
            session.run(
                "MERGE (a:Subscriber {id: $caller}) " +
                "MERGE (b:Subscriber {id: $callee}) " +
                "MERGE (a)-[r:CALL {duration: $dur}]->(b)",
                Values.parameters("caller", op.caller, "callee", op.callee, "dur", op.duration)
            );
        } catch (Exception e) {
            logger.error("Neo4j write error", e);
        }
    }
}
