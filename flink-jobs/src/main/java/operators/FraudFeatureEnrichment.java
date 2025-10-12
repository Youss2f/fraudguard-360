package operators;

import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import com.fraudguard.models.GraphOperation;
import models.CDR;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FraudFeatureEnrichment extends KeyedProcessFunction<String, String, GraphOperation> {
    private static final Logger logger = LoggerFactory.getLogger(FraudFeatureEnrichment.class);

    private transient ValueState<Integer> callCountState;

    @Override
    public void open(Configuration parameters) {
        ValueStateDescriptor<Integer> descriptor = new ValueStateDescriptor<>("callCount", org.apache.flink.api.common.typeinfo.Types.INT);
        callCountState = getRuntimeContext().getState(descriptor);
    }

    @Override
    public void processElement(String json, Context ctx, Collector<GraphOperation> out) throws Exception {
        try {
            CDR cdr = new ObjectMapper().readValue(json, CDR.class);
            int currentCount = callCountState.value() + 1;
            callCountState.update(currentCount);
            // Enrich: Add features if needed
            out.collect(new GraphOperation(cdr.getCallerId(), cdr.getCalleeId(), cdr.getDuration()));
        } catch (Exception e) {
            logger.error("Processing error", e);
        }
    }
}
