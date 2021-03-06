package gfar;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.Reranker;
import gfar.rerank.*;
import gfar.util.LoadData;
import gfar.util.Params;
import gfar.util.ParseArgs;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
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
public class RunGreedyAlgorithms {
    public static void main(String[] args) throws Exception {
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
                    Map<Long, List<Long>> groups = LoadData.loadGroups(groupsFilePath);
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

                            rerankersMap.put("GreedyLM" + "_" + lambda, () -> new GreedyLM<>(lambda, cutoff, maxLength, true, groups, individualRecommendations, "BORDA"));
                            rerankersMap.put("SPGreedy" + "_" + lambda, () -> new SPGreedy<>(lambda, cutoff, maxLength, true, groups, individualRecommendations, 1, 1));

                        }
                        rerankersMap.forEach(Unchecked.biConsumer((name, rerankerSupplier) -> {
                            System.out.println("Running " + name);
                            String recOut = DATA_PATH + fold + "/" + fileName + "_" + name;
                            Reranker<Long, Long> reranker = rerankerSupplier.get();
                            try (RecommendationFormat.Writer<Long, Long> writer = format.getWriter(recOut)) {
                                format.getReader(groupRecIn).readAll()
                                        .map(rec -> reranker.rerankRecommendation(new Recommendation<>(rec.getUser(), rec.getItems()), maxLength))
                                        .forEach(Unchecked.consumer(writer::write));
                            }
                        }));
                    }
                }
            }
        }
    }
}
