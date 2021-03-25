package gfar;

import es.uam.eps.ir.ranksys.core.Recommendation;
import gfar.aggregation.AverageAggregationStrategyPref;
import gfar.aggregation.AverageScoreAggregationStrategy;
import gfar.aggregation.FairnessAggregationStrategy;
import gfar.aggregation.MaxSatisfactionAggStrategy;
import gfar.aggregation.XPO;
import gfar.util.LoadData;
import gfar.util.Params;
import gfar.util.ParseArgs;

import org.javatuples.Pair;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import java.io.IOException;
import java.util.*;

import static org.ranksys.formats.parsing.Parsers.lp;
import java.nio.file.Paths;

/**
 * After generating the individual recommendations (ordered sets) for each
 * individual, compute the group recommendations using AVG, FAI and XPO by using
 * this file.
 */

public class AggregateRecommedations {
    public static void main(String[] args) throws Exception {
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();

        String[] strategies = { "XPO", "FAI", "AVG", "AVG20" };

        Params params = ParseArgs.Parse(args);
        if (params == null) {
            System.exit(1);
        }

        final boolean runUserPref = params.allParams.contains("--userPref");
        if (runUserPref) {
            strategies = new String[] { "AVG20" };
        }

        for (int i = 0; i < params.datasets.size(); i++) {
            String DATA_PATH = PROJECT_FOLDER + "/data/" + params.datasets.get(i) + "/";
            String fileName = params.individualRecFileName.get(i);
            for (String fold : params.folds) {
                System.out.println("fold: " + fold);
                Map<Long, Recommendation<Long, Long>> recommendation = new HashMap<>();
                RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<Long, Long>(lp, lp);
                String recIn = DATA_PATH + fold + "/" + fileName;
                format.getReader(recIn).readAll().forEach(rec -> {
                    recommendation.put(rec.getUser(), rec);
                });
                for (int groupSize : params.groupSize) {
                    System.out.println("group size: " + groupSize);
                    for (String groupType : params.groupTypes) {
                        System.out.println("group type: " + groupType);
                        String filePath = DATA_PATH + groupType + "_group_" + groupSize;

                        for (String strategy : strategies) {
                            System.out.println(strategy);
                            if (runUserPref) {
                                String out_file = DATA_PATH + fold + "/" + fileName + "_" + strategy.toLowerCase() + "_"
                                        + groupType + "_group_" + groupSize + "_weighted";
                                Map<Long, Pair<List<Long>, List<Double>>> groups = LoadData
                                        .loadGroupsWithUserPreferences(filePath);
                                computeWeightedGroupRecs(out_file, groups, recommendation, format);

                            } else {
                                String out_file = DATA_PATH + fold + "/" + fileName + "_" + strategy.toLowerCase() + "_"
                                        + groupType + "_group_" + groupSize;
                                Map<Long, List<Long>> groups = LoadData.loadGroups(filePath);
                                computeGroupRecs(out_file, strategy, groups, recommendation, format);
                            }
                        }
                    }
                }
            }
        }
    }

    public static void computeGroupRecs(String outFile, String strategy, Map<Long, List<Long>> groups,
            Map<Long, Recommendation<Long, Long>> recommendation, RecommendationFormat<Long, Long> format) {

        try {
            RecommendationFormat.Writer<Long, Long> writer = format.getWriter(outFile);
            groups.forEach((gID, members) -> {
                Recommendation<Long, Long> group_recs = null;
                switch (strategy) {
                case "AVG":
                    group_recs = (new AverageScoreAggregationStrategy<Long, Long, Long>(gID, members, 10000)).aggregate(recommendation);
                    break;
                case "AVG20":
                    group_recs = (new AverageScoreAggregationStrategy<Long, Long, Long>(gID, members, 20)).aggregate(recommendation);
                    break;
                case "FAI":
                    group_recs = (new FairnessAggregationStrategy<Long, Long, Long>(gID, members, 20)).aggregate(recommendation);
                    break;
                case "XPO":
                    group_recs = (new XPO<Long, Long, Long>(gID, members, 20)).aggregate(recommendation);
                    break;
                case "MAX":
                    group_recs = (new MaxSatisfactionAggStrategy<Long, Long, Long>(gID, members, 20).aggregate(recommendation));
                    break;
                }

                // Here write top-N recs for the group to a file!
                writer.accept(group_recs);
            });
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void computeWeightedGroupRecs(String outFile, Map<Long, Pair<List<Long>, List<Double>>> groups,
            Map<Long, Recommendation<Long, Long>> recommendation, RecommendationFormat<Long, Long> format) {
        try {
            RecommendationFormat.Writer<Long, Long> writer = format.getWriter(outFile);
            groups.forEach((gID, members) -> {
                Recommendation<Long, Long> group_recs = new AverageAggregationStrategyPref<Long, Long, Long>(gID,
                        members, 20, null).aggregate(recommendation);
                // Here write top-N recs for the group to a file!
                writer.accept(group_recs);
            });
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
