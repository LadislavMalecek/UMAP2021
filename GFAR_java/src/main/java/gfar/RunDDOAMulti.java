package gfar;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.Reranker;
import gfar.rerank.FuzzyDHondt;
import gfar.rerank.FuzzyDHondtDirectOptimize;
import gfar.util.*;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import org.jooq.lambda.Unchecked;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Supplier;

import static org.ranksys.formats.parsing.Parsers.lp;

/**
 *
 */
public class RunDDOAMulti {
    public static void main(String[] args) throws Exception {
        System.out.println("Running DHondstAlgorithms");
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();

        Params params = ParseArgs.Parse(args);
        if (params == null) {
            System.exit(1);
        }

        double[] lambdas = { 1.0 };
        int cutoff = 10000;
        int maxLength = 20;

        for (int i = 0; i < params.datasets.size(); i++) {
            String DATA_PATH = PROJECT_FOLDER + "/data/" + params.datasets.get(i) + "/";
            String individualRecsFileName = params.individualRecFileName.get(i);

            for (int size : params.groupSize) {
                System.out.println("Group Size: " + size);
                for (String groupType : params.groupTypes) {
                    System.out.println("Group Type: " + groupType);
                    String fileName = individualRecsFileName + "_avg_" + groupType + "_group_" + size;
                    String groupsFilePath = DATA_PATH + groupType + "_group_" + size;
                    Map<Long, List<Long>> groups = loadGroups(groupsFilePath);
                    for (String fold : params.folds) {
                        System.out.println("Fold: " + fold);
                        String recIn = DATA_PATH + fold + "/" + individualRecsFileName;
                        String groupRecIn = DATA_PATH + fold + "/" + fileName;

                        RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<Long, Long>(lp, lp);
                        Map<Long, Recommendation<Long, Long>> individualRecommendations = new HashMap<>();

                        Map<String, Supplier<Reranker<Long, Long>>> rerankersMap = new HashMap<>();
                        System.out.println("Reading the individual score");
                        format.getReader(recIn).readAll().forEach(rec -> {
                            Object2DoubleOpenHashMap<Long> user_predictions = new Object2DoubleOpenHashMap<>();
                            user_predictions.defaultReturnValue(0.0);
                            rec.getItems().forEach(longTuple2od -> {
                                user_predictions.addTo(longTuple2od.v1, longTuple2od.v2);
                            });

                            individualRecommendations.put(rec.getUser(), rec);
                        });
                        for (double lambda : lambdas) {
                            Double minLimit = params.datasets.get(i).contains("ml1m") ? 4.0 : 1.0;

                            rerankersMap.put("FuzzyDHondtDirectOptimizeLimit" + minLimit + "_" + lambda,
                                    () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                            individualRecommendations, /* individualPreferences */ null,
                                            /* minimumItemScoreToBeConsidered */ minLimit, /* constantDecrease */ null,
                                            /* negativePartMultiplier */ null, /* discountByPossition */ false,
                                            /* excessMultiplier */ null, null));

                            rerankersMap.put("FuzzyDHondtDirectOptimizeRelu0.5" + "_" + lambda,
                                    () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                            individualRecommendations, /* individualPreferences */ null,
                                            /* minimumItemScoreToBeConsidered */ null, /* constantDecrease */ null,
                                            /* negativePartMultiplier */ 1.0 / 2, /* discountByPossition */ false,
                                            /* excessMultiplier */ null, null));

                            rerankersMap.put("FuzzyDHondtDirectOptimizeRelu0.125" + "_" + lambda,
                                    () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                            individualRecommendations, /* individualPreferences */ null,
                                            /* minimumItemScoreToBeConsidered */ null, /* constantDecrease */ null,
                                            /* negativePartMultiplier */ 1.0 / 8, /* discountByPossition */ false,
                                            /* excessMultiplier */ null, null));

                            rerankersMap.put("FuzzyDHondtDirectOptimizePosDiscount" + "_" + lambda,
                                    () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                            individualRecommendations, /* individualPreferences */ null,
                                            /* minimumItemScoreToBeConsidered */ null, /* constantDecrease */ null,
                                            /* negativePartMultiplier */ null, /* discountByPossition */ true,
                                            /* excessMultiplier */ null, null));

                            rerankersMap.put("FuzzyDHondtDirectOptimizeConstDec4" + "_" + lambda,
                                    () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                            individualRecommendations, /* individualPreferences */ null,
                                            /* minimumItemScoreToBeConsidered */ null, /* constantDecrease */ 4.0,
                                            /* negativePartMultiplier */ null, /* discountByPossition */ false,
                                            /* excessMultiplier */ null, null));

                            rerankersMap.put("FuzzyDHondtDirectOptimizeConstDec3.5" + "_" + lambda,
                                    () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                            individualRecommendations, /* individualPreferences */ null,
                                            /* minimumItemScoreToBeConsidered */ null, /* constantDecrease */ 3.5,
                                            /* negativePartMultiplier */ null, /* discountByPossition */ false,
                                            /* excessMultiplier */ null, null));

                            rerankersMap.put("FuzzyDHondtDirectOptimizeExcessMul0.2" + "_" + lambda,
                                    () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                            individualRecommendations, /* individualPreferences */ null,
                                            /* minimumItemScoreToBeConsidered */ null, /* constantDecrease */ 3.5,
                                            /* negativePartMultiplier */ null, /* discountByPossition */ false,
                                            /* excessMultiplier */ 0.2, null));

                            rerankersMap.put("FuzzyDHondtDirectOptimizeSquare" + "_" + lambda,
                                    () -> new FuzzyDHondtDirectOptimize<>(lambda, cutoff, true, maxLength, groups,
                                            individualRecommendations, /* individualPreferences */ null,
                                            /* minimumItemScoreToBeConsidered */ null, /* constantDecrease */ 3.5,
                                            /* negativePartMultiplier */ null, /* discountByPossition */ false,
                                            /* excessMultiplier */ null, 2.0));

                        }
                        rerankersMap.forEach(Unchecked.biConsumer((name, rerankerSupplier) -> {
                            System.out.println("Running " + name);
                            String recOut = DATA_PATH + fold + "/" + fileName + "_" + name;
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

    /**
     * Loads the ids of the users for each group from a file (for synthetic groups)!
     *
     * @param filePath
     * @return
     */
    public static Map<Long, List<Long>> loadGroups(String filePath) {
        Scanner s = null;
        try {
            s = new Scanner(new File(filePath));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        Map<Long, List<Long>> groups = new HashMap<>();

        if (s != null) {
            while (s.hasNext()) {
                List<Long> group_members = new ArrayList<>();
                String[] parsedLine = s.nextLine().split("\t");
                long id = Long.parseLong(parsedLine[0]);
                for (int i = 1; i < parsedLine.length; i++) {
                    group_members.add(Long.parseLong(parsedLine[i]));
                }
                groups.put(id, group_members);
            }
        }
        if (s != null) {
            s.close();
        }
        return groups;
    }
}
