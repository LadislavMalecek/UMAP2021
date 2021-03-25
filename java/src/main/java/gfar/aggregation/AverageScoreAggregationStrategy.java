package gfar.aggregation;
import java.util.List;

/**
 * Average aggregation strategy (AVG in the paper). If some items do not exist
 * for some users it assigns score 0 to them Notice that, if we generate scores
 * for individuals based on their unseen items it is possible to recommend to a
 * group that is seen by some of the group members!
 *
 * @param <U>
 * @param <I>
 * @param <G>
 */
public class AverageScoreAggregationStrategy<U, I, G> extends SimpleAbstractAggregationStrategy<U, I, G> {
    public AverageScoreAggregationStrategy(G groupID, List<U> group_members, int top_N) {
        super(groupID, group_members, top_N);
    }

    @Override
    protected double aggregateOldWithNewValue(double currentValue, I item, double itemScore) {
        // counting mean when knowing # members in advance
        // sum(x)/n == sum(x/n)
        return currentValue + (itemScore / this.group_members.size());
    }

}
