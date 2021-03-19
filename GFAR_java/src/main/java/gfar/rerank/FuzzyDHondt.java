package gfar.rerank;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.LambdaReranker;

import org.javatuples.Pair;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * SPGreedy algorithm from the paper "Fairness in Package-to-Group
 * Recommendations", it uses m-proportionality! In our paper the results are for
 * m=1 (single proportionality), the code allows to experiment with different
 * values of m!
 *
 *
 * @param <G>
 * @param <U>
 * @param <I>
 */
public class FuzzyDHondt<G, U, I> extends LambdaReranker<G, I> {

    // Algorithm's parameters

    // Data for recommender
    protected Map<G, Pair<List<U>, List<Double>>> groupMembers;
    protected int maxLength;
    protected Map<U, Recommendation<U, I>> individualRecommendations;

    // used to return relevance of recommended items per users
    protected Map<G, List<Double>> returnGroupItemsRelevance;

    // rearanged recommandations to allow for mor efficient data access
    protected Map<U, Map<I, Double>> recommendations;

    public FuzzyDHondt(Double lambda, int cutoff, boolean norm, int maxLength,
        Map<G, Pair<List<U>, List<Double>>> groupMembers, Map<U, Recommendation<U, I>> individualRecommendations,
        Double minimumItemScoreToBeConsidered, Double constantDecrease, Double negativePartMultiplier,
        Double exponentialFactor){
        this(lambda,  cutoff, norm, maxLength, groupMembers, individualRecommendations, minimumItemScoreToBeConsidered,
        constantDecrease, negativePartMultiplier, exponentialFactor, null);
    }

    public FuzzyDHondt(double lambda, int cutoff, boolean norm, int maxLength,
            Map<G, Pair<List<U>, List<Double>>> groupMembers, Map<U, Recommendation<U, I>> individualRecommendations,
            Double minimumItemScoreToBeConsidered, Double constantDecrease, Double negativePartMultiplier,
            Double exponentialFactor, Map<G, List<Double>> returnGroupItemsRelevance) {
        super(lambda, cutoff, norm);
        System.out.println("FuzzyDHondt constructor");
        this.groupMembers = groupMembers;
        this.individualRecommendations = individualRecommendations;
        this.maxLength = maxLength;

        this.TransformRecommendations(minimumItemScoreToBeConsidered, constantDecrease, negativePartMultiplier,
                exponentialFactor);

        this.returnGroupItemsRelevance = returnGroupItemsRelevance;
    }

    // Transforms recommendations into sensible data access first format
    private void TransformRecommendations(Double minimumItemScoreToBeConsidered, Double constantDecrease,
            Double negativePartMultiplier, Double exponentialFactor) {
        System.out.println("TransformRecommendations started.");
        this.recommendations = new HashMap<>();
        for (Recommendation<U, I> recommendations : individualRecommendations.values()) {
            U user = recommendations.getUser();
            Map<I, Double> userMap = new HashMap<>();
            for (Tuple2od<I> recommendation : recommendations.getItems()) {
                Double rating = recommendation.v2;

                if (constantDecrease != null) {
                    rating -= constantDecrease;
                }

                if (rating < 0 && negativePartMultiplier != null) {
                    rating *= negativePartMultiplier;
                }

                if (minimumItemScoreToBeConsidered != null && recommendation.v2 < minimumItemScoreToBeConsidered) {
                    // skip adding the value - so when retrieved it will be the default 0
                    continue;
                }
                if (exponentialFactor != null && rating > 0) {
                    rating = Math.pow(rating, exponentialFactor);
                }
                userMap.put(recommendation.v1, recommendation.v2);
            }
            this.recommendations.put(user, userMap);
        }
        System.out.println("TransformRecommendations finished.");
    }

    @Override
    protected GreedyUserReranker<G, I> getUserReranker(Recommendation<G, I> recommendation, int maxLength) {
        return new UserFuzzyDHondt(recommendation, maxLength);
    }

    public class UserFuzzyDHondt extends LambdaUserReranker {
        private final G group;

        // Dhondt's voting at the beginning of the algoritm
        // starting DHondt's votes, stays fixed

        // current voting power, as fraction of groupIndividualPreferences and
        // groupAlreadyUsedSelectionPower
        // by the Dhondt's rule (fuzzy)

        // order of users is the same
        private List<U> usersInGroup;
        private List<Double> usersPrefInGroup;

        private Double[] startingVotingSupport;
        private Double[] currentVotingSupportToUsers;
        // sum of relevance of already selected items to each user
        private Double[] alreadySelectedItemsRelevanceToUsers;

        /**
         * Constructor.
         *
         * @param recommendation        input recommendation
         * @param individualPreferences starting prefferences to users in reranking
         * @param maxLength             maximum length of the re-ranked recommendation
         */
        public UserFuzzyDHondt(Recommendation<G, I> recommendation, int maxLength) {
            super(recommendation, maxLength);
            System.out.println("UserFuzzyDHondtDirect constructor");

            this.group = recommendation.getUser();
            this.usersInGroup = groupMembers.get(this.group).getValue0();
            this.usersPrefInGroup = groupMembers.get(this.group).getValue1();
            int groupSize = this.usersInGroup.size();

            // initialize the preferences, if pref included in data then use them, if not
            // use uniform 1
            boolean prefdataMissing = this.usersPrefInGroup == null || this.usersPrefInGroup.isEmpty();
            if (prefdataMissing) {
                this.startingVotingSupport = new Double[groupSize];
                Arrays.fill(this.startingVotingSupport, 1.0);
                this.currentVotingSupportToUsers = new Double[groupSize];
                Arrays.fill(this.currentVotingSupportToUsers, 1.0);
            } else {
                this.startingVotingSupport = usersPrefInGroup.toArray(new Double[groupSize]);
                this.currentVotingSupportToUsers = usersPrefInGroup.toArray(new Double[groupSize]);
            }

            this.alreadySelectedItemsRelevanceToUsers = new Double[groupSize];
            Arrays.fill(this.alreadySelectedItemsRelevanceToUsers, 1.0);
        }

        private void recalculateCurrentDHnodtsSupport() {
            for (int uIndex = 0; uIndex < this.usersInGroup.size(); uIndex++) {
                currentVotingSupportToUsers[uIndex] = startingVotingSupport[uIndex]
                        / alreadySelectedItemsRelevanceToUsers[uIndex];
            }
        }

        private Double[] getRecommendationsForUsers(List<U> users, I item) {
            Double[] recommendationsArray = new Double[users.size()];

            int uIndex = 0;
            for (U user : usersInGroup) {
                Map<I, Double> usersRecommendations = recommendations.get(user);
                if (usersRecommendations == null) {
                    recommendationsArray[uIndex] = 0.0;
                    continue;
                }
                recommendationsArray[uIndex] = usersRecommendations.getOrDefault(item, 0.0);
                uIndex++;
            }
            return recommendationsArray;
        }

        @Override
        protected double nov(Tuple2od<I> tupleOfItemAndValue) {
            // for each user check if recommending the item
            // if true then add its score multiplied by users current voting power to items
            // utility
            I item = tupleOfItemAndValue.v1;
            Double itemsUtility = 0.0;

            Double[] recommendationsForUsers = getRecommendationsForUsers(usersInGroup, item);
            for (int uIndex = 0; uIndex < usersInGroup.size(); uIndex++) {
                Double relevanceToUser = recommendationsForUsers[uIndex];
                itemsUtility += relevanceToUser * this.currentVotingSupportToUsers[uIndex];
            }

            return itemsUtility;
        }

        @Override
        protected void update(Tuple2od<I> bestItemValue) {
            System.out.println("Object selected: " + bestItemValue.v1);
            // Update individual utility for each group member here, after selecting a new
            // item greedily!
            // check who voted for the selected user
            // for each user that did, decrease its voting power by the fuzzy DHondt's rule
            I item = bestItemValue.v1;
            Double[] recommendationsForUser = getRecommendationsForUsers(usersInGroup, item);
            for (int uIndex = 0; uIndex < usersInGroup.size(); uIndex++) {
                // check if item is relevant to this user
                Double itemRelevanceToUser = recommendationsForUser[uIndex];
                // if it is, then update (add that value) alreadySelectedRelevance for this user
                this.alreadySelectedItemsRelevanceToUsers[uIndex] += itemRelevanceToUser;
            }

            this.recalculateCurrentDHnodtsSupport();

            // we could do this also in discounted fassion
            if (returnGroupItemsRelevance != null) {
                Double sum = Arrays.asList(this.alreadySelectedItemsRelevanceToUsers).stream().mapToDouble(m -> m)
                        .sum();
                List<Double> normalized = Arrays.asList(this.alreadySelectedItemsRelevanceToUsers).stream()
                        .map(m -> m / sum).collect(Collectors.toList());
                returnGroupItemsRelevance.put(group, normalized);
            }
        }
    }
}
