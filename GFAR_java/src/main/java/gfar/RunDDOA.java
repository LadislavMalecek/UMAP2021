package gfar;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.Reranker;
import gfar.rerank.FuzzyDHondt;
import gfar.rerank.FuzzyDHondtDirectOptimize;
import gfar.util.*;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;

import org.javatuples.Pair;
import org.jooq.lambda.Unchecked;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import java.nio.file.Paths;
import java.util.*;
import java.util.function.Supplier;

import static org.ranksys.formats.parsing.Parsers.lp;

/**
 *
 */
public class RunDDOA {
    public static void main(String[] args) throws Exception {
        System.out.println("Running DHondstAlgorithms");
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();

        Params params = ParseArgs.Parse(args);
        if (params == null) {
            System.exit(1);
        }

        final boolean runExtended = params.allParams.contains("--extended");
        final boolean runUserPref = params.allParams.contains("--userPref");

        double[] lambdas = { 1.0 };
        int cutoff = 10000;
        int maxLength = 20;

        for (int i = 0; i < params.datasets.size(); i++) {
            String DATA_PATH = PROJECT_FOLDER + "/data/" + params.datasets.get(i) + "/";
            String individualRecsFileName = params.individualRecFileName.get(i);
            for (String fold : params.folds) {
                System.out.println("Fold: " + fold);
                String recIn = DATA_PATH + fold + "/" + individualRecsFileName;
                
                // load main prefference data for the current fold
                System.out.println("Reading the individual score");
                Map<Long, Recommendation<Long, Long>> individualRecommendations = new HashMap<>();
                RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<Long, Long>(lp, lp);
                format.getReader(recIn).readAll().forEach(rec -> {
                    Object2DoubleOpenHashMap<Long> user_predictions = new Object2DoubleOpenHashMap<>();
                    user_predictions.defaultReturnValue(0.0);
                    rec.getItems().forEach(longTuple2od -> {
                        user_predictions.addTo(longTuple2od.v1, longTuple2od.v2);
                    });
                    
                    individualRecommendations.put(rec.getUser(), rec);
                });
                
                Map<String, Supplier<Reranker<Long, Long>>> rerankersMap = new HashMap<>();
                for (int size : params.groupSize) {
                    System.out.println("Group Size: " + size);
                    for (String groupType : params.groupTypes) {
                        System.out.println("Group Type: " + groupType);
                        String fileName = individualRecsFileName + "_avg_" + groupType + "_group_" + size;

                        // Load group data and preferences if specified
                        String groupRecIn = DATA_PATH + fold + "/" + fileName;
                        String groupsFilePath = DATA_PATH + groupType + "_group_" + size;
                        Map<Long, Pair<List<Long>, List<Double>>> groups = runUserPref ?
                            LoadData.loadGroupsWithUserPreferences(groupsFilePath) :
                            LoadData.loadGroupsWithUniformUserPreferences(groupsFilePath, 1.0);

                        for (double lambda : lambdas) {
                            rerankersMap.put("FuzzyDHondt" + "_" + lambda, () -> new FuzzyDHondt<>(lambda, cutoff, true,
                                    maxLength, groups, individualRecommendations, null, null, null, null));

                            rerankersMap.put("FuzzyDHondtDirectOptimize" + "_" + lambda,
                                    () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                            individualRecommendations, null, null, null, false, null, null));

                            if (runExtended) {
                                Double minLimit = params.datasets.get(i).contains("ml1m") ? 4.0 : 1.0;
                                rerankersMap.put("FuzzyDHondtDirectOptimizeLimit" + minLimit + "_" + lambda,
                                        () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                                individualRecommendations,
                                                /* minimumItemScoreToBeConsidered */ minLimit,
                                                /* constantDecrease */ null, /* negativePartMultiplier */ null,
                                                /* discountByPossition */ false, /* excessMultiplier */ null, null));

                                rerankersMap.put("FuzzyDHondtDirectOptimizeRelu0.5" + "_" + lambda,
                                        () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                                individualRecommendations, /* minimumItemScoreToBeConsidered */ null,
                                                /* constantDecrease */ null, /* negativePartMultiplier */ 1.0 / 2,
                                                /* discountByPossition */ false, /* excessMultiplier */ null, null));

                                rerankersMap.put("FuzzyDHondtDirectOptimizeRelu0.125" + "_" + lambda,
                                        () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                                individualRecommendations, /* minimumItemScoreToBeConsidered */ null,
                                                /* constantDecrease */ null, /* negativePartMultiplier */ 1.0 / 8,
                                                /* discountByPossition */ false, /* excessMultiplier */ null, null));

                                rerankersMap.put("FuzzyDHondtDirectOptimizePosDiscount" + "_" + lambda,
                                        () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                                individualRecommendations, /* minimumItemScoreToBeConsidered */ null,
                                                /* constantDecrease */ null, /* negativePartMultiplier */ null,
                                                /* discountByPossition */ true, /* excessMultiplier */ null, null));

                                rerankersMap.put("FuzzyDHondtDirectOptimizeConstDec4" + "_" + lambda,
                                        () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                                individualRecommendations, /* minimumItemScoreToBeConsidered */ null,
                                                /* constantDecrease */ 4.0, /* negativePartMultiplier */ null,
                                                /* discountByPossition */ false, /* excessMultiplier */ null, null));

                                rerankersMap.put("FuzzyDHondtDirectOptimizeConstDec3.5" + "_" + lambda,
                                        () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                                individualRecommendations, /* minimumItemScoreToBeConsidered */ null,
                                                /* constantDecrease */ 3.5, /* negativePartMultiplier */ null,
                                                /* discountByPossition */ false, /* excessMultiplier */ null, null));

                                rerankersMap.put("FuzzyDHondtDirectOptimizeExcessMul0.2" + "_" + lambda,
                                        () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                                individualRecommendations, /* minimumItemScoreToBeConsidered */ null,
                                                /* constantDecrease */ null, /* negativePartMultiplier */ null,
                                                /* discountByPossition */ false, /* excessMultiplier */ 0.2, null));

                                rerankersMap.put("FuzzyDHondtDirectOptimizeExponential2.0" + "_" + lambda,
                                        () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                                individualRecommendations, /* minimumItemScoreToBeConsidered */ null,
                                                /* constantDecrease */ null, /* negativePartMultiplier */ null,
                                                /* discountByPossition */ false, /* excessMultiplier */ null, 2.0));

                                rerankersMap.put("FuzzyDHondtDirectOptimizeExp2CDec3Relu0.25Exc0.25" + "_" + lambda,
                                        () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                                individualRecommendations, /* minimumItemScoreToBeConsidered */ null,
                                                /* constantDecrease */ 3.0, /* negativePartMultiplier */ 1.0 / 4,
                                                /* discountByPossition */ false, /* excessMultiplier */ 1.0 / 4, 
                                                /* exponenial */ 2.0));
                            }
                        }
                        rerankersMap.forEach(Unchecked.biConsumer((name, rerankerSupplier) -> {
                            System.out.println("Running " + name);
                            String recOut = DATA_PATH + fold + "/" + fileName + "_" + name;
                            if(runUserPref){
                                recOut = DATA_PATH + fold + "/" + fileName + "_" + name + "_weighted";
                            }
                            System.out.println(recOut);
                            Reranker<Long, Long> reranker = rerankerSupplier.get();
                            try (RecommendationFormat.Writer<Long, Long> writer = format.getWriter(recOut)) {
                                format.getReader(groupRecIn).readAll()
                                        .map(rec -> reranker.rerankRecommendation(
                                                new Recommendation<>(rec.getUser(), rec.getItems()), maxLength))
                                        .forEach(Unchecked.consumer(writer::write));
                            }
                        }));
                    }
                }
            }
        }
    }
}
