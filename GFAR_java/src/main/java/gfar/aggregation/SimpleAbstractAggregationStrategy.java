package gfar.aggregation;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.ranksys.core.util.tuples.Tuple2od;

import es.uam.eps.ir.ranksys.core.Recommendation;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;

/**
 * Abstract aggregation strategy. Future common functionalities for aggregation
 * strategies can be implemented here
 * 
 * @param <U>
 * @param <I>
 * @param <G>
 */
public abstract class SimpleAbstractAggregationStrategy<U, I, G> implements AggregationStrategy<U, I, G> {
    protected G groupID;
    protected List<U> group_members;
    protected long top_N;

    protected SimpleAbstractAggregationStrategy(G groupID, List<U> group_members, int top_N) {
        this.groupID = groupID;
        this.group_members = group_members;
        this.top_N = top_N;
    }

    protected abstract double aggregateOldWithNewValue(double currentValue, I item, double itemScore);

    @Override
    public Recommendation<G, I> aggregate(Map<U, Recommendation<U, I>> recommendations) {

        Object2DoubleOpenHashMap<I> minScores = new Object2DoubleOpenHashMap<>();
        minScores.defaultReturnValue(0.0);

        for (U user : group_members) {
            if (!recommendations.containsKey(user)) {
                System.out.println(groupID);
                continue;
            }
            for (Tuple2od<I> recommendation : recommendations.get(user).getItems()) {
                I item = recommendation.v1;
                double score = recommendation.v2;
                // aggregation strategy
                minScores.compute(item, (itemKey, currentValue) -> aggregateOldWithNewValue((currentValue == null? 0.0 : currentValue), item, score));
            }
        }
        Comparator<Tuple2od<I>> sortingComparator = Comparator.comparingDouble((Tuple2od<I> r) -> r.v2).reversed();
        List<Tuple2od<I>> topNScores = minScores.entrySet().stream()
                .map(entry -> new Tuple2od<I>(entry.getKey(), entry.getValue())).sorted(sortingComparator).limit(top_N)
                .collect(Collectors.toList());

        return new Recommendation<G, I>(groupID, topNScores);
    }

}
