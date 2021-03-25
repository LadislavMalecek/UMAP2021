package gfar.aggregation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.ranksys.core.util.tuples.Tuple2od;

import es.uam.eps.ir.ranksys.core.Recommendation;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;

import org.javatuples.*;

/**
 * Abstract aggregation strategy. Future common functionalities for aggregation
 * strategies can be implemented here
 * 
 * @param <U>
 * @param <I>
 * @param <G>
 */
public class AverageAggregationStrategyPref<U, I, G> implements AggregationStrategy<U, I, G> {
    protected G groupID;
    protected List<U> usersInGroup;
    protected List<Double> usersPrefInGroup;
    protected long top_N;

    Map<G, List<Double>> returnGroupItemsRelevance;

    public AverageAggregationStrategyPref(G groupID, Pair<List<U>, List<Double>> group_members, int top_N, Map<G, List<Double>> returnGroupItemsRelevance) {
        this.groupID = groupID;
        this.usersInGroup = group_members.getValue0();
        this.usersPrefInGroup = group_members.getValue1();

        this.top_N = top_N;
        this.returnGroupItemsRelevance = returnGroupItemsRelevance;
    }

    @Override
    public Recommendation<G, I> aggregate(Map<U, Recommendation<U, I>> recommendations) {

        Object2DoubleOpenHashMap<I> scores = new Object2DoubleOpenHashMap<>();
        scores.defaultReturnValue(0.0);

        Map<I, List<Double>> itemsScores = new HashMap<>();

        int index = 0;
        for (U user : usersInGroup) {
            Double usersPref = this.usersPrefInGroup.get(index);
            if (!recommendations.containsKey(user)) {
                System.out.println(groupID);
                continue;
            }
            for (Tuple2od<I> recommendation : recommendations.get(user).getItems()) {
                I item = recommendation.v1;
                double score = recommendation.v2 * usersPref;
                // aggregation strategy
                scores.compute(item, (itemKey, currentValue) -> (currentValue == null? 0.0 : currentValue)  + score);

                List<Double> list = itemsScores.getOrDefault(item, null);
                if(list == null){
                    list = new ArrayList<>(Collections.nCopies(usersInGroup.size(), 0.0));
                }
                list.set(index, recommendation.v2);
                itemsScores.put(item, list);
            }
            index++;
        }
        Comparator<Tuple2od<I>> sortingComparator = Comparator.comparingDouble((Tuple2od<I> r) -> r.v2).reversed();
        List<Tuple2od<I>> topNScores = scores.entrySet().stream()
                .map(entry -> new Tuple2od<I>(entry.getKey(), entry.getValue())).sorted(sortingComparator).limit(top_N)
                .collect(Collectors.toList());

        if(returnGroupItemsRelevance != null){
            List<Double> usersRel = new ArrayList<>(Collections.nCopies(usersInGroup.size(), 0.0));

            for (Tuple2od<I> recommendation : topNScores) {
                List<Double> itemRelToUsers = itemsScores.get(recommendation.v1);
                for (int userIndex = 0; userIndex < usersInGroup.size(); userIndex++) {
                    Double itemScoreForUser = itemRelToUsers.get(userIndex);
                    usersRel.set(userIndex, usersRel.get(userIndex) + itemScoreForUser);
                }
            }
            Double sum = usersRel.stream().mapToDouble(m -> m).sum();
            List<Double> normalized = usersRel.stream().map(m -> m / sum).collect(Collectors.toList());
            returnGroupItemsRelevance.put(this.groupID, normalized);
        }

        return new Recommendation<G, I>(groupID, topNScores);
    }
}
