package gfar.metrics;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.AbstractRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.basic.NDCG;
import es.uam.eps.ir.ranksys.metrics.rank.RankingDiscountModel;
import gfar.util.ListAnalyzer;


import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * NDCG metric in the paper. NDCG (min or minmax).
 *
 * @param <G>
 * @param <I>
 * @param <U>
 */
public class RelevanceSumGroupFairness<G, I, U> extends AbstractRecommendationMetric<G, I> {

    private final RelevanceModel<U, I> relModel;
    private final int cutoff;
    private final Map<G, List<U>> groups;
    private final String groupAgregType;

    public RelevanceSumGroupFairness(RelevanceModel<U, I> relModel, int cutoff, Map<G, List<U>> groups, String groupArgegType) {
        this.relModel = relModel;
        this.cutoff = cutoff;
        this.groups = groups;
        this.groupAgregType = groupArgegType;
    }

    @Override
    public double evaluate(Recommendation<G, I> recommendation) {
        List<U> groupMembers = groups.get(recommendation.getUser());
        List<Double> group_val = new ArrayList<>();
        for (U user : groupMembers) {
            RelevanceModel<U, I>.UserRelevanceModel userRelModel = (RelevanceModel<U, I>.UserRelevanceModel) relModel.getModel(user);
            double sum = recommendation.getItems().stream()
                    .limit(cutoff)
                    .mapToDouble(pair -> userRelModel
                    .gain(pair.v1))
                    .sum();
            group_val.add(sum);
        }

        return ListAnalyzer.Eval(group_val, groupAgregType);
    }
}
