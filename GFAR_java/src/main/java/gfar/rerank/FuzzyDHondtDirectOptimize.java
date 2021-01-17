package gfar.rerank;

import es.uam.eps.ir.ranksys.core.Recommendation;
import org.ranksys.core.util.tuples.Tuple2od;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
public class FuzzyDHondtDirectOptimize<G, U, I> extends FuzzyDHondt<G, U, I> {
    private Boolean isDiscountByPossition;
    private Double excessTaperMultiplier;
    
    public FuzzyDHondtDirectOptimize(double lambda, int cutoff, boolean norm, int maxLength,
            Map<G, List<U>> group_members, Map<U, Recommendation<U, I>> individualRecommendations,
            Map<U, Double> individualPreferences, Double minimumItemScoreToBeConsidered, Double constantDecrease,
            Double negativePartMultiplier, Boolean discountByPossition, Double excessMultiplier) {

        super(lambda, cutoff, norm, maxLength, group_members, individualRecommendations, individualPreferences,
                minimumItemScoreToBeConsidered, constantDecrease, negativePartMultiplier);
            
        this.isDiscountByPossition = discountByPossition;
        this.excessTaperMultiplier = excessMultiplier;
        System.out.println("FuzzyDHondtDirectOptimizeGreedy constructor");
        System.out.println("Limit set to: " + minimumItemScoreToBeConsidered);
    }

    @Override
    protected GreedyUserReranker<G, I> getUserReranker(Recommendation<G, I> recommendation, int maxLength) {
        return new UserFuzzyDHondtDirectOptimize(recommendation, individualPreferences, maxLength, this.isDiscountByPossition, this.excessTaperMultiplier);
    }

    // runs for each group
    public class UserFuzzyDHondtDirectOptimize extends LambdaUserReranker {
        private final G group;

        private Map<U, Double> currentRelevanceOfSelectedForUsers;

        private Double totalUtilityForSelectedSoFar;

        private List<U> usersInGroup;

        private Boolean discountByPossition;
        private int currentPosition = 1;
        
        private double currentMultiplier = 1.0;
        
        //by default we cut it away
        private double excessMultiplier = 0.0;
        /**
         * Constructor.
         *
         * @param recommendation        input recommendation
         * @param individualPreferences starting prefferences to users in reranking
         * @param maxLength             maximum length of the re-ranked recommendation
         */
        public UserFuzzyDHondtDirectOptimize(Recommendation<G, I> recommendation, Map<U, Double> individualPreferences,
                int maxLength, Boolean isDiscountByPossition, Double excessTaperMultiplier) {
            super(recommendation, maxLength);
            System.out.println("UserFuzzyDHondtDirectOptimize constructor");
            this.group = recommendation.getUser();
            this.usersInGroup = group_members.get(this.group);

            // this.partyVotingSupport = individualPreferences;
            // this.normalizePartyVotingSupport();

            this.currentRelevanceOfSelectedForUsers = new HashMap<>();

            this.totalUtilityForSelectedSoFar = 0.0;
            this.discountByPossition = isDiscountByPossition;

            if(excessTaperMultiplier != null){
                this.excessMultiplier = excessTaperMultiplier.doubleValue();
            }
        }

        private Double getSumOfRelevanceOfSelectedForUsers() {
            return this.currentRelevanceOfSelectedForUsers.values().stream().reduce(0.0, Double::sum);
        }

        private double[] getRecommendationsForUsers(List<U> users, I item) {
            double[] recommendationsArray = new double[users.size()];

            int index = 0;
            for (U user : usersInGroup) {
                Map<I, Double> usersRecommendations = recommendations.get(user);
                if (usersRecommendations == null) {
                    recommendationsArray[index] = 0.0;
                    continue;
                }
                recommendationsArray[index] = usersRecommendations.getOrDefault(item, 0.0) * this.currentMultiplier;
                index++;
            }
            return recommendationsArray;
        }

        private double totalUtility(double[] recommendations) {
            double sum = 0.0;
            for (double utility : recommendations) {
                sum += utility;
            }
            return sum;
        }

        @Override
        protected double nov(Tuple2od<I> tupleOfItemAndValue) {
            // for each user check if recommending the item
            // if true then add its score multiplied by users current voting power to items
            // utility
            I item = tupleOfItemAndValue.v1;
            double itemsUtility = 0.0;

            // Double itemsCeiledUtility = 0.0;


            // used as a temporary storage so that we do not query the data twice
            double[] recommendationsForUser = getRecommendationsForUsers(usersInGroup, item);
            double itemsTotalUtility = totalUtility(recommendationsForUser);

            double totalPlusProspectedUtility = itemsTotalUtility + totalUtilityForSelectedSoFar;

            double uniformVotingSupport = 1.0 / usersInGroup.size();
            int index = 0;
            for (U user : usersInGroup) {
                // get the fraction of under-representation for the party
                // check how much proportional representation the candidate adds
                // sum over all parties & select the highest sum

                // Double usersSupport = this.partyVotingSupport.getOrDefault(user, 0.0);
                double totalPlusProspectedUtilityScaled = totalPlusProspectedUtility * uniformVotingSupport;

                double alreadySelectedRelevanceToParty = currentRelevanceOfSelectedForUsers.getOrDefault(user, 0.0);

                // how much relevance can we add before we hit the current ceiling
                double notFulfilledPartyRelevance = Double.max(0,
                        totalPlusProspectedUtilityScaled - alreadySelectedRelevanceToParty);

                double itemsUnscaledRelevance = recommendationsForUser[index];

                // itemsUtility += Double.min(itemsUnscaledRelevance, notFulfilledPartyRelevance);

                double amountOverShot = itemsUnscaledRelevance - notFulfilledPartyRelevance;
                if(amountOverShot > 0){
                    itemsUtility += notFulfilledPartyRelevance + ( amountOverShot * this.excessMultiplier);
                } else {
                    itemsUtility += itemsUnscaledRelevance;
                }
                

                index++;
            }

            return itemsUtility;
        }

        @Override
        protected void update(Tuple2od<I> bestItemValue) {
            // Update individual utility for each group member here, after selecting a new
            // item greedily!
            // check who voted for the selected user
            // for each user that did, decrease its voting power by the fuzzy DHondt's rule
            I item = bestItemValue.v1;
            List<U> usersInGroup = group_members.get(this.group);

            double[] recommendationsForUsers = getRecommendationsForUsers(usersInGroup, item);

            double totalUtilityForNewItem = 0.0;

            int index = 0;
            for (U user : usersInGroup) {
                // check if item is relevant to this user
                double itemRelevanceToUser = recommendationsForUsers[index];
                double currentRelevanceOfSelectedForUser = this.currentRelevanceOfSelectedForUsers.getOrDefault(user, 0.0);
                double newRelevanceOfSelectedForUser = currentRelevanceOfSelectedForUser + itemRelevanceToUser;
                this.currentRelevanceOfSelectedForUsers.put(user, newRelevanceOfSelectedForUser);

                totalUtilityForNewItem += itemRelevanceToUser;

                index++;
            }

            System.out.println("Object selected: " + bestItemValue.v1 + " utility: " + totalUtilityForNewItem
                    + " individualUtil: " + Arrays.toString(recommendationsForUsers) + " curMult: " + this.currentMultiplier);

            this.totalUtilityForSelectedSoFar = this.getSumOfRelevanceOfSelectedForUsers();

            this.currentPosition++;

            if(this.discountByPossition){
                // includes change of bases from natural to base 2
                // possition starts at 1 and we want 1/dcg
                // this.currentMultiplier = 1 / (Math.log(2.0) / Math.log(this.currentPosition + 1)); is eq with below
                this.currentMultiplier = Math.log(2.0) / Math.log(this.currentPosition + 1);
            }
        }
    }
}
