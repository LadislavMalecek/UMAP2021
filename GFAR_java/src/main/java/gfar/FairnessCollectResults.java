package gfar;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.metrics.RecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.SystemMetric;
import es.uam.eps.ir.ranksys.metrics.basic.AverageRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.rank.LogarithmicDiscountModel;
import es.uam.eps.ir.ranksys.metrics.rel.BinaryRelevanceModel;

import gfar.metrics.*;
import gfar.util.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import org.javatuples.Pair;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import static org.ranksys.formats.parsing.Parsers.lp;

public class FairnessCollectResults {

    static List<String> algorithmNames = new ArrayList<>(
            Arrays.asList("AVG", "FAI", "XPO", "GreedyLM", "SPGreedy", "GFAR"));
    static List<String> reRankStrings = new ArrayList<>(
            Arrays.asList("", "", "", "_GreedyLM_1.0", "_SPGreedy_1.0", "_GFAR_1.0"));

    private static void initAlgNames(Boolean extended) {
        algorithmNames.add("FuzzyDHondt");
        reRankStrings.add("_FuzzyDHondt_1.0");

        algorithmNames.add("FuzzyDHondtDirectOptimize");
        reRankStrings.add("_FuzzyDHondtDirectOptimize_1.0");

        if (extended) {

            algorithmNames.add("FuzzyDHondtDirectOptimizeLimit4.0");
            reRankStrings.add("_FuzzyDHondtDirectOptimizeLimit4.0_1.0");

            algorithmNames.add("FuzzyDHondtDirectOptimizeRelu0.5");
            reRankStrings.add("_FuzzyDHondtDirectOptimizeRelu0.5_1.0");

            algorithmNames.add("FuzzyDHondtDirectOptimizeRelu0.125");
            reRankStrings.add("_FuzzyDHondtDirectOptimizeRelu0.125_1.0");

            algorithmNames.add("FuzzyDHondtDirectOptimizePosDiscount");
            reRankStrings.add("_FuzzyDHondtDirectOptimizePosDiscount_1.0");

            algorithmNames.add("FuzzyDHondtDirectOptimizeConstDec4");
            reRankStrings.add("_FuzzyDHondtDirectOptimizeConstDec4_1.0");

            algorithmNames.add("FuzzyDHondtDirectOptimizeConstDec3.5");
            reRankStrings.add("_FuzzyDHondtDirectOptimizeConstDec3.5_1.0");

            algorithmNames.add("FuzzyDHondtDirectOptimizeExcessMul0.2");
            reRankStrings.add("_FuzzyDHondtDirectOptimizeExcessMul0.2_1.0");
        }
    }

    public static void main(String[] args) throws Exception {
        Boolean extended = Arrays.asList(args).contains("--extended");
        initAlgNames(extended);

        Params params = ParseArgs.Parse(args);
        if (params == null) {
            System.exit(1);
        }

        params.datasets = new ArrayList<>(Arrays.asList("ml1m"));

        
        String PROJECT_FOLDER = Paths.get(System.getProperty("user.dir")).getParent().toString();


        Boolean useTrainingSet = Arrays.asList(args).contains("--useTrainingSet");
        if(useTrainingSet){
            System.out.println("=================================================");
            System.out.println("=================================================");
            System.out.println("=================================================");
            System.out.println("Results collected using training&test set");
            System.out.println("=================================================");
            System.out.println("=================================================");
            System.out.println("=================================================");
        }

        Boolean useMFSet = Arrays.asList(args).contains("--useMFSet");
        if (useTrainingSet) {
            System.out.println("=================================================");
            System.out.println("=================================================");
            System.out.println("=================================================");
            System.out.println("Results collected using training&test set");
            System.out.println("=================================================");
            System.out.println("=================================================");
            System.out.println("=================================================");
        }

        for (String dataset : params.datasets) {
            System.out.println("Results for the " + dataset + " dataset");
            System.out.println("=================================================");
            String DATA_PATH = PROJECT_FOLDER + "/data/" + dataset + "/";

            int cutoff = 20;
            double relevanceThreshold = dataset.equals("kgrec") ? 1.0 : 4.0;
            String baseFileName = dataset.equals("kgrec") ? "mf_230_1.0" : "mf_30_1.0";

            String fileNameBase;

            LogarithmicDiscountModel ldisc = new LogarithmicDiscountModel();

            String testFileUsed = "test.csv";
            if (useTrainingSet) {
                testFileUsed = "train.csv";
            }

            if (useMFSet) {
                testFileUsed = baseFileName + ".csv";
            }

            System.out.println("Test file used: " + testFileUsed);

            for (String groupType : params.groupTypes) {
                System.out.println("Results for group Type: " + groupType);

                Boolean headingPrinted = false;
                for (int groupSize : params.groupSize) {
                    for (int i = 0; i < algorithmNames.size(); i++) {
                        String algorithm = algorithmNames.get(i);
                        switch (algorithm) {
                            case "LM":
                                fileNameBase = baseFileName + "_lm_" + groupType + "_group_";
                                break;
                            case "FAI":
                                fileNameBase = baseFileName + "_fai_" + groupType + "_group_";
                                break;
                            case "XPO":
                                fileNameBase = baseFileName + "_xpo_" + groupType + "_group_";
                                break;
                            case "MAX":
                                fileNameBase = baseFileName + "_max_" + groupType + "_group_";
                                break;
                            default:
                                fileNameBase = baseFileName + "_avg_" + groupType + "_group_";
                                break;
                        }
                        String reRank = reRankStrings.get(i);
                        Map<String,List<Double>> results = new HashMap<>();
                        List<String> metrics = new ArrayList<>();

                        Boolean metricsInitialized = false;
                    
                        String fileName = fileNameBase + groupSize;
                        String groupsFilePath = DATA_PATH + groupType + "_group_" + groupSize;
                        Map<Long, List<Long>> groups = loadGroups(groupsFilePath);
                        int numGroups = groups.size();


                        for (String fold : params.folds) {
                            String testDataPath = DATA_PATH + fold + "/" + testFileUsed;
                            
                            PreferenceData<Long, Long> testData = SimplePreferenceData.load(GFARPreferenceReader.get().read(testDataPath, lp, lp));
                            BinaryRelevanceModel<Long, Long> binRel = new BinaryRelevanceModel<>(false, testData, relevanceThreshold);
                            RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<Long, Long>(lp, lp);
                            String recIn = DATA_PATH + fold + "/" + fileName + reRank;
                            List<Pair<String, SystemMetric<Long, Long>>> sysMetrics = new ArrayList<>();
                            List<Pair<String, RecommendationMetric<Long, Long>>> recMetrics = new ArrayList<>();

                            // Zero Recall Metric
                            // recMetrics.add(new Pair<>("recallzero", new RecallGroupFairness<>(cutoff, groups, binRel, "ZERO")));

                            // // Recall Metrics (mean, min, minmax)
                            // recMetrics.add(new Pair<>("recall", new RecallGroupFairness<>(cutoff, groups, binRel, "AVG")));
                            // recMetrics.add(new Pair<>("recallmin", new RecallGroupFairness<>(cutoff, groups, binRel, "MIN")));
                            // recMetrics.add(new Pair<>("recallminmax", new RecallGroupFairness<>(cutoff, groups, binRel, "MIN-MAX")));
                            // recMetrics.add(new Pair<>("recallstd", new RecallGroupFairness<>(cutoff, groups, binRel, "STD")));

                            recMetrics.add(new Pair<>("precision", new PrecisionGroup<>(cutoff, groups, binRel, "AVG")));
                            recMetrics.add(new Pair<>("precisionmin", new PrecisionGroup<>(cutoff, groups, binRel, "MIN")));
                            recMetrics.add(new Pair<>("precisionminmax", new PrecisionGroup<>(cutoff, groups, binRel, "MIN-MAX")));
                            recMetrics.add(new Pair<>("precisionstd", new PrecisionGroup<>(cutoff, groups, binRel, "STD")));

                            // //NDCG Metrics (mean, min, minmax)
                            // recMetrics.add(new Pair<>("ndcg", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "EXP"), cutoff, ldisc, groups, "AVG")));
                            // recMetrics.add(new Pair<>("ndcgmin", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "EXP"), cutoff, ldisc, groups, "MIN")));
                            // recMetrics.add(new Pair<>("ndcgminmax", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "EXP"), cutoff, ldisc, groups, "MIN-MAX")));
                            // recMetrics.add(new Pair<>("ndcgstd", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "EXP"), cutoff, ldisc, groups, "STD")));


                            // // NDCG Metrics (mean, min, minmax) without treshold
                            // recMetrics.add(new Pair<>("ndcg_treshold_0", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, 0, "EXP"), cutoff, ldisc, groups, "AVG")));
                            // recMetrics.add(new Pair<>("ndcgmin_treshold_0", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, 0, "EXP"), cutoff, ldisc, groups, "MIN")));
                            // recMetrics.add(new Pair<>("ndcgminmax_treshold_0", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, 0, "EXP"), cutoff, ldisc, groups, "MIN-MAX")));
                            // recMetrics.add(new Pair<>("ndcgstd_treshold_0", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, 0, "EXP"), cutoff, ldisc, groups, "STD")));



                            // LINEAR RELEVANCE GAIN
                            // NDCG Metrics (mean, min, minmax)
                            recMetrics.add(new Pair<>("ndcg_linear_gain", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "LIN"), cutoff, ldisc, groups, "AVG")));
                            recMetrics.add(new Pair<>("ndcgmin_linear_gain", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "LIN"), cutoff, ldisc, groups, "MIN")));
                            recMetrics.add(new Pair<>("ndcgminmax_linear_gain", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "LIN"), cutoff, ldisc, groups, "MIN-MAX")));
                            recMetrics.add(new Pair<>("ndcg_linear_gain_std", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "LIN"), cutoff, ldisc, groups, "STD")));

                            // LINEAR RELEVANCE GAIN
                            // NDCG Metrics (mean, min, minmax) without treshold
                            recMetrics.add(new Pair<>("ndcg_linear_gain_treshold_0", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, 0, "LIN"), cutoff, ldisc, groups, "AVG")));
                            recMetrics.add(new Pair<>("ndcgmin_linear_gain_treshold_0", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, 0, "LIN"), cutoff, ldisc, groups, "MIN")));
                            recMetrics.add(new Pair<>("ndcgminmax_linear_gain_treshold_0", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, 0, "LIN"), cutoff, ldisc, groups, "MIN-MAX")));
                            recMetrics.add(new Pair<>("ndcgstd_linear_gain_treshold_0", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, 0, "LIN"), cutoff, ldisc, groups, "STD")));


                            // BINAR RELEVANCE GAIN
                            // NDCG Metrics (mean, min, minmax)
                            recMetrics.add(new Pair<>("ndcg_bin_gain", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "BIN"), cutoff, ldisc, groups, "AVG")));
                            recMetrics.add(new Pair<>("ndcgmin_bin_gain", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "BIN"), cutoff, ldisc, groups, "MIN")));
                            recMetrics.add(new Pair<>("ndcgminmax_bin_gain", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "BIN"), cutoff, ldisc, groups, "MIN-MAX")));
                            recMetrics.add(new Pair<>("ndcg_bin_gain_std", new NDCGGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "BIN"), cutoff, ldisc, groups, "STD")));


                            // relevance sum
                            // recMetrics.add(new Pair<>("relevancesum", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "EXP"), cutoff, groups, "AVG")));
                            // recMetrics.add(new Pair<>("relevancesummin", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "EXP"), cutoff, groups, "MIN")));
                            // recMetrics.add(new Pair<>("relevancesumminmax", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "EXP"), cutoff, groups, "MIN-MAX")));
                            // recMetrics.add(new Pair<>("relevancesumstd", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "EXP"), cutoff, groups, "STD")));

                            // recMetrics.add(new Pair<>("relevancesum_treshold_0", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, 0, "EXP"), cutoff, groups, "AVG")));
                            // recMetrics.add(new Pair<>("relevancesummin_treshold_0", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, 0, "EXP"), cutoff, groups, "MIN")));
                            // recMetrics.add(new Pair<>("relevancesumminmax_treshold_0", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, 0, "EXP"), cutoff, groups, "MIN-MAX")));
                            // recMetrics.add(new Pair<>("relevancesumstd_treshold_0", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, 0, "EXP"), cutoff, groups, "STD")));


                            // relevance sum
                            recMetrics.add(new Pair<>("relevancesum_linear_gain", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "LIN"), cutoff, groups, "AVG")));
                            recMetrics.add(new Pair<>("relevancesummin_linear_gain", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "LIN"), cutoff, groups, "MIN")));
                            recMetrics.add(new Pair<>("relevancesumminmax_linear_gain", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "LIN"), cutoff, groups, "MIN-MAX")));
                            recMetrics.add(new Pair<>("relevancesumstd_linear_gain", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, relevanceThreshold, "LIN"), cutoff, groups, "STD")));

                            recMetrics.add(new Pair<>("relevancesum_linear_gain_treshold_0", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, 0, "LIN"), cutoff, groups, "AVG")));
                            recMetrics.add(new Pair<>("relevancesummin_linear_gain_treshold_0", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, 0, "LIN"), cutoff, groups, "MIN")));
                            recMetrics.add(new Pair<>("relevancesumminmax_linear_gain_treshold_0", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, 0, "LIN"), cutoff, groups, "MIN-MAX")));
                            recMetrics.add(new Pair<>("relevancesumstd_linear_gain_treshold_0", new RelevanceSumGroupFairness<>(new RelevanceModel<>(false, testData, 0, "LIN"), cutoff, groups, "STD")));


                            // DFH Metrics (mean, min, minmax)
                            recMetrics.add(new Pair<>("dfh", new DiscountedFirstHitFairness<>(cutoff, ldisc, groups, binRel, "AVG")));
                            recMetrics.add(new Pair<>("dfhmin", new DiscountedFirstHitFairness<>(cutoff, ldisc, groups, binRel, "MIN")));
                            recMetrics.add(new Pair<>("dfhminmax", new DiscountedFirstHitFairness<>(cutoff, ldisc, groups, binRel, "MIN-MAX")));
                            recMetrics.add(new Pair<>("dfhstd", new DiscountedFirstHitFairness<>(cutoff, ldisc, groups, binRel, "STD")));

                            recMetrics.forEach(pair -> sysMetrics.add(new Pair<>(pair.getValue0(), new AverageRecommendationMetric<>(pair.getValue1(), true))));

                            // format.getReader(recIn).readAll().forEach(rec -> sysMetrics.values().forEach(metric -> metric.add(rec)));
                            for(Recommendation<Long, Long> recommendation : format.getReader(recIn).readAll().collect(Collectors.toList())){
                                // put in the test set
                                for(Pair<String, SystemMetric<Long, Long>> sysMetric : sysMetrics){
                                    sysMetric.getValue1().add(recommendation);
                                }
                            }
                            
                            // eval the metrics
                            for(Pair<String, SystemMetric<Long, Long>> sysMetric : sysMetrics){
                                String name = sysMetric.getValue0();
                                SystemMetric<Long, Long> metric = sysMetric.getValue1();
    
                                results.computeIfAbsent(name, k -> new ArrayList<>()).add(metric.evaluate());
                                if(!metricsInitialized){
                                    metrics.add(name);
                                }
                            }
                            metricsInitialized = true;

                        } // folds end
                        

                        if(!headingPrinted){
                            List<String> heading = new ArrayList<String>();
                            heading.add("algorithm");
                            heading.add("groupSize");
                            metrics.forEach(metric ->{
                                heading.add(metric);
                                // heading.add(metric + "_std");
                            });
                            System.out.println(String.join(",", heading));
                            headingPrinted = true;
                        }
                        
                        // // print results
                        List<String> individualResults = new ArrayList<>();
                        for(String metric : metrics){
                            Double result = results.get(metric).stream().mapToDouble(r -> r).average().orElse(0.0);
                            individualResults.add(String.valueOf(result));
                        }
                        String outStr = algorithm + "," + groupSize + "," + String.join(",", individualResults);
                        System.out.println(outStr);
                    }
                }
                System.out.println("=================================================");
            }
            System.out.println("=================================================");
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
