package gfar.metrics;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.metrics.rel.IdealRelevanceModel;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;

/**
 * Relevance model for nDCG, in which the gains of all relevant documents need
 * to be known for the normalization of the metric.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 */
public class RelevanceModel<U, I> extends IdealRelevanceModel<U, I> {

    private final PreferenceData<U, I> testData;
    private final double threshold;
    private final String type;

    /**
     * Constructors.
     *
     * @param caching   are the user relevance models being cached?
     * @param testData  test subset of preferences
     * @param threshold relevance threshold
     */
    public RelevanceModel(boolean caching, PreferenceData<U, I> testData, double threshold, String type) {
        super(caching, testData.getUsersWithPreferences());
        this.testData = testData;
        this.threshold = threshold;
        this.type = type;
    }

    /**
     * {@inheritDoc}
     *
     * @param user input user
     * @return ndcg relevance model for input user
     */
    @Override
    protected UserRelevanceModel get(U user) {
        return new UserRelevanceModel(user);
    }

    /**
     * User relevance model for nDCG.
     *
     */
    public class UserRelevanceModel implements IdealRelevanceModel.UserIdealRelevanceModel<U, I> {

        private final Object2DoubleMap<I> gainMap;

        /**
         * Constructor.
         *
         * @param user user whose relevance model is computed
         */
        public UserRelevanceModel(U user) {
            this.gainMap = new Object2DoubleOpenHashMap<>();
            gainMap.defaultReturnValue(0.0);

            if (type == "LIN") {
                List<IdPref<I>> userRec = testData.getUserPreferences(user).filter(iv -> iv.v2 >= threshold)
                        .collect(Collectors.toList());
                for (IdPref<I> rec : userRec) {
                    gainMap.put(rec.v1, rec.v2 - threshold + 1);
                }

                // testData.getUserPreferences(user).filter(iv -> iv.v2 >= threshold)
                //         .forEach(iv -> gainMap.put(iv.v1, iv.v2 - threshold + 1));
            } else if (type == "EXP") {
                testData.getUserPreferences(user).filter(iv -> iv.v2 >= threshold)
                        .forEach(iv -> gainMap.put(iv.v1, Math.pow(2, iv.v2 - threshold + 1.0) - 1.0));
            } else if (type == "BIN") {
                testData.getUserPreferences(user).filter(iv -> iv.v2 >= threshold)
                        .forEach(iv -> gainMap.put(iv.v1, 1.0));
            } else
                throw new IllegalArgumentException("Invalid relevance type: " + type);
        }

        /**
         * {@inheritDoc}
         *
         * @return set of relevant items
         */
        @Override
        public Set<I> getRelevantItems() {
            return gainMap.keySet();
        }

        /**
         * {@inheritDoc}
         *
         * @param item input item
         * @return true is item is relevant, false otherwise
         */
        @Override
        public boolean isRelevant(I item) {
            return gainMap.containsKey(item);
        }

        /**
         * {@inheritDoc}
         *
         * @param item input item
         * @return relevance gain of the input item
         */
        @Override
        public double gain(I item) {
            return gainMap.getDouble(item);
        }

        /**
         * Get the vector of gains of the relevant items.
         *
         * @return array of positive relevance gains
         */
        public double[] getGainValues() {
            return gainMap.values().toDoubleArray();
        }
    }
}