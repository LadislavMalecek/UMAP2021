package gfar;

import es.uam.eps.ir.ranksys.core.Recommendation;
import gfar.aggregation.AverageAggregationStrategyPref;
import gfar.util.Params;
import gfar.util.ParseArgs;

import org.javatuples.Pair;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

import static org.ranksys.formats.parsing.Parsers.lp;
import java.nio.file.Paths;

/**
 * After generating the individual recommendations (ordered sets) for each individual, compute the group
 * recommendations using AVG, FAI and XPO by using this file.
 */

public class AggregateRecommedationsPref {
    public static void main(String[] args) throws Exception {
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();
        Params params = ParseArgs.Parse(args);
        if (params == null) {
            System.exit(1);
        }

        params.datasets = Arrays.asList("ml1m");
        
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

                        Map<Long, Pair<List<Long>, List<Double>>> groups = loadGroups(filePath);

                        String out_file = DATA_PATH + fold + "/" + fileName + "_avg_" + groupType + "_group_" + groupSize + "_pref_AVG";

                        computeGroupRecs(out_file, groups, recommendation, format);
                        
                    }
                }
            }
        }
    }

    public static void computeGroupRecs(String outFile, Map<Long, Pair<List<Long>, List<Double>>> groups,
                                        Map<Long, Recommendation<Long, Long>> recommendation,
                                        RecommendationFormat<Long, Long> format) {
        try {
            RecommendationFormat.Writer<Long, Long> writer = format.getWriter(outFile);
            groups.forEach((gID, members) ->  {
                Recommendation<Long, Long> group_recs = new AverageAggregationStrategyPref<Long, Long, Long>(gID, members, 20, null).aggregate(recommendation);
                // Here write top-N recs for the group to a file!
                writer.accept(group_recs);
            });
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads the ids of the users for each group from a file (for synthetic groups)!
     *
     * @param filePath
     * @return
     */
    public static Map<Long, Pair<List<Long>, List<Double>>> loadGroups(String filePath) {
        Scanner s = null;
        Scanner s_pref = null;
        try {
            s = new Scanner(new File(filePath));
            s_pref = new Scanner(new File(filePath + "_weights"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        Map<Long, Pair<List<Long>, List<Double>>> groups = new HashMap<>();

        if (s != null) {
            while (s.hasNext()) {
                List<Long> group_members = new ArrayList<>();
                List<Double> group_prefs = new ArrayList<>();
                String[] parsedLine = s.nextLine().split("\t");
                long id = Long.parseLong(parsedLine[0]);

                String[] parsedLinePref = s_pref.nextLine().split("\t");
                long id_pref = Long.parseLong(parsedLine[0]);

                if(id != id_pref){
                    throw new IllegalStateException("Error: " + filePath);
                }

                for (int i = 1; i < parsedLine.length; i++) {
                    group_members.add(Long.parseLong(parsedLine[i]));
                    group_prefs.add(Double.parseDouble(parsedLinePref[i]));
                }
                groups.put(id, new Pair<>(group_members, group_prefs));
            }
        }
        if (s != null) {
            s.close();
        }
        return groups;
    }
}
