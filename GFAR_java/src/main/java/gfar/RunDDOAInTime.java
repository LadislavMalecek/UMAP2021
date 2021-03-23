package gfar;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.novdiv.reranking.Reranker;
import gfar.aggregation.AverageAggregationStrategyPref;
import gfar.rerank.FuzzyDHondt;
import gfar.rerank.FuzzyDHondtDirectOptimize;
import gfar.util.*;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;

import org.javatuples.Pair;
import org.jooq.lambda.Unchecked;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import static org.ranksys.formats.parsing.Parsers.lp;

/**
 *
 */
public class RunDDOAInTime {
    public static void main(String[] args) throws Exception {
        System.out.println("Running DHondstAlgorithms");
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();

        Params params = ParseArgs.Parse(args);
        if (params == null) {
            System.exit(1);
        }

        params.datasets = Arrays.asList("ml1m");
        params.individualRecFileName = Arrays.asList("mf_30_1.0");

        int cutoff = 10000;
        int maxLength = 20;

        for (String alg : Arrays.asList("default", "directOptimize", "avg")) {
            for (int i = 0; i < params.datasets.size(); i++) {
                String DATA_PATH = PROJECT_FOLDER + "/data/" + params.datasets.get(i) + "/";
                String individualRecsFileName = params.individualRecFileName.get(i);

                Map<String, Supplier<Reranker<Long, Long>>> rerankersMap = new HashMap<>();
                for (int size : params.groupSize) {
                    System.out.println("Group Size: " + size);
                    for (String groupType : params.groupTypes) {
                        System.out.println("Group Type: " + groupType);
                        String fileName = individualRecsFileName + "_avg_" + groupType + "_group_" + size;

                        List<Map<Long, List<Double>>> listOfPreviousSatisfactions = new ArrayList<>();

                        for (String fold : params.folds) {
                            Map<Long, List<Double>> currentRunEndGroupUsersPref = new HashMap<>();

                            System.out.println("Fold: " + fold);
                            String recIn = DATA_PATH + fold + "/" + individualRecsFileName;

                            // load main prefference data for the current fold
                            System.out.println("Reading the individual score");
                            Map<Long, Recommendation<Long, Long>> individualRecommendations = new HashMap<>();
                            RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<Long, Long>(lp,
                                    lp);
                            format.getReader(recIn).readAll().forEach(rec -> {
                                Object2DoubleOpenHashMap<Long> user_predictions = new Object2DoubleOpenHashMap<>();
                                user_predictions.defaultReturnValue(0.0);
                                rec.getItems().forEach(longTuple2od -> {
                                    user_predictions.addTo(longTuple2od.v1, longTuple2od.v2);
                                });

                                individualRecommendations.put(rec.getUser(), rec);
                            });

                            // Load group data and preferences if specified
                            String groupRecIn = DATA_PATH + fold + "/" + fileName;
                            String groupsFilePath = DATA_PATH + groupType + "_group_" + size;

                            Map<Long, Pair<List<Long>, List<Double>>> groups;
                            if (fold == "1") {
                                groups = LoadData.loadGroupsWithUniformUserPreferences(groupsFilePath, 1.0);
                            } else {
                                Map<Long, List<Double>> curPrefs = getCurrentGroupPref(listOfPreviousSatisfactions,
                                        size);
                                groups = LoadData.loadGroupsWithUserPreferencesFromMap(groupsFilePath, curPrefs);
                            }

                            String algName = "";

                            if (alg == "default") {
                                algName = "FuzzyDHondt_longterm";
                                rerankersMap.put(algName,
                                        () -> new FuzzyDHondt<>(1.0, cutoff, true, maxLength, groups,
                                                individualRecommendations, null, null, null, null,
                                                currentRunEndGroupUsersPref));

                            } else if (alg == "directOptimize") {
                                algName = "FuzzyDHondtDirectOptimize_longterm";
                                rerankersMap.put(algName,
                                        () -> new FuzzyDHondtDirectOptimize<>(1.0, cutoff, true, maxLength, groups,
                                                individualRecommendations, null, null, null, false, null, null,
                                                currentRunEndGroupUsersPref));
                            } else if (alg == "avg") {
                                algName = "AVG_longterm";
                                String recOut = DATA_PATH + fold + "/" + fileName + "_" + algName;
                                computeGroupRecs(recOut, groups, individualRecommendations, format, currentRunEndGroupUsersPref);
                            } else {
                                throw new IllegalArgumentException("invalid alg");
                            }

                            if(alg != "avg"){
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

                            listOfPreviousSatisfactions.add(currentRunEndGroupUsersPref);
                            String recRelOut = DATA_PATH + fold + "/" + fileName + "_" + algName + "_rec_rel";
                            saveRecRels(currentRunEndGroupUsersPref, recRelOut);
                        }
                    }
                }
            }
        }

    }

    public static void computeGroupRecs(String outFile, Map<Long, Pair<List<Long>, List<Double>>> groups,
            Map<Long, Recommendation<Long, Long>> recommendation, RecommendationFormat<Long, Long> format, 
            Map<Long, List<Double>> returnGroupItemsRelevance) {
        try {
            RecommendationFormat.Writer<Long, Long> writer = format.getWriter(outFile);
            groups.forEach((gID, members) -> {
                Recommendation<Long, Long> group_recs = new AverageAggregationStrategyPref<Long, Long, Long>(gID,
                        members, 20, returnGroupItemsRelevance).aggregate(recommendation);
                // Here write top-N recs for the group to a file!
                writer.accept(group_recs);
            });
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Map<Long, List<Double>> getCurrentGroupPref(List<Map<Long, List<Double>>> runs, int groupSize) {
        Map<Long, List<Double>> sumOfPrefsForRunsSoFar = new HashMap<>();
        int numRunsSoFar = runs.size();

        for (Map<Long, List<Double>> run : runs) {
            for (Map.Entry<Long, List<Double>> entry : run.entrySet()) {
                // for each group in a singe run
                Long groupId = entry.getKey();
                List<Double> runGroupPrefs = entry.getValue();

                // now I want to update the sumOfPrefsForRunsSoFar map entry of that group with
                // online mean computation
                List<Double> runsToDateGroupPrefsSums = sumOfPrefsForRunsSoFar.getOrDefault(groupId,
                        new ArrayList<Double>(Collections.nCopies(groupSize, 0.0)));

                List<Double> newReturnGroupPrefs = new ArrayList<Double>();
                for (int index = 0; index < runGroupPrefs.size(); index++) {
                    Double current = runsToDateGroupPrefsSums.get(index);
                    Double newValue = runGroupPrefs.get(index);

                    newReturnGroupPrefs.add(current + newValue);
                }

                sumOfPrefsForRunsSoFar.put(groupId, newReturnGroupPrefs);
            }
        }
        Map<Long, List<Double>> nextRunPref = new HashMap<>();
        Double eachMemberShouldMaxRecieve = (numRunsSoFar + 1) / (double) groupSize;
        for (Map.Entry<Long, List<Double>> entry : sumOfPrefsForRunsSoFar.entrySet()) {
            List<Double> groupsNextPref = new ArrayList<Double>();
            for (Double sumForGroupMember : entry.getValue()) {
                groupsNextPref.add(Math.max(0, eachMemberShouldMaxRecieve - sumForGroupMember));
            }
            nextRunPref.put(entry.getKey(), groupsNextPref);
        }

        return nextRunPref;
    }

    private static void saveRecRels(Map<Long, List<Double>> runRecRel, String file) throws IOException {

        FileWriter writer = new FileWriter(file);

        List<Map.Entry<Long, List<Double>>> list = new ArrayList<>(runRecRel.entrySet());
        list.sort(Comparator.comparingLong(Map.Entry::getKey));
        for (Map.Entry<Long, List<Double>> entry : list) {
            List<String> values = entry.getValue().stream().map(d -> d.toString()).collect(Collectors.toList());
            String line = entry.getKey() + "\t" + String.join("\t", values);
            writer.write(line + System.lineSeparator());
        }
        writer.close();
    }
}
