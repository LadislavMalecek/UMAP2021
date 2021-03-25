package gfar.aggregation;
import java.util.*;

public class MaxSatisfactionAggStrategy<U, I, G> extends SimpleAbstractAggregationStrategy<U, I, G> {
    public MaxSatisfactionAggStrategy(G groupID, List<U> group_members, int top_N){
        super(groupID, group_members, top_N);
    }

    @Override
    protected double aggregateOldWithNewValue(double currentValue, I item, double itemScore) {
        return Math.max(currentValue, itemScore);
    }
}
