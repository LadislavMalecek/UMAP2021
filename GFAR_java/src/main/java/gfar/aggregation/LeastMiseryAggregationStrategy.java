package gfar.aggregation;
import java.util.*;

/**
 * Least misery score aggregation strategy. If some items do not exist for some
 * users it assigns score 0 to them. Notice that, if we generate scores for
 * individuals based on their unseen items it is possible to recommend to a
 * group that is seen by some of the group members!
 *
 * @param <U>
 * @param <I>
 * @param <G>
 */
public class LeastMiseryAggregationStrategy<U, I, G> extends SimpleAbstractAggregationStrategy<U, I, G> {

    public LeastMiseryAggregationStrategy(G groupID, List<U> group_members, int top_N) {
        super(groupID, group_members, top_N);
    }

    @Override
    protected double aggregateOldWithNewValue(double currentValue, I item, double itemScore) {
        return Math.min(currentValue, itemScore);
    }
}
