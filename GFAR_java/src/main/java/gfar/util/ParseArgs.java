package gfar.util;

import java.util.ArrayList;
import java.util.Arrays;

public class ParseArgs {
    public static Params Parse(String[] args){

        final String algPrefix = "--data=";
        final String foldsPrefix = "--folds=";
        final String groupSizePrefix = "--groupSize=";
        final String groupTypePrefix = "--groupType=";
        final String help = "--help";
        final String test = "--test";

        Params params = new Params();

        if(Arrays.asList(args).contains(test)){
            params.datasets = Arrays.asList("ml1m");
            params.individualRecFileName = Arrays.asList("mf_30_1.0.test");
            params.folds = Arrays.asList("1");
            params.groupSize = Arrays.asList(3);
            params.groupTypes = Arrays.asList("sim");
            return params;
        }

        if (Arrays.asList(args).contains(help)) {
            System.out.println("Possible params: " + algPrefix + " " + foldsPrefix + " " + groupSizePrefix + " "
                    + groupTypePrefix);
            return null;
        }
        
        for(String arg:args)
        {
            if (arg.startsWith(algPrefix)) {
                
                params.datasets = new ArrayList<>();
                params.individualRecFileName = new ArrayList<>();
                
                String alg = arg.substring(algPrefix.length());
                if (alg.contains("ml1m")) {
                    params.datasets.add("ml1m");
                    params.individualRecFileName.add("mf_30_1.0");
                }
                if (alg.contains("kgrec")) {
                    params.datasets.add("kgrec");
                    params.individualRecFileName.add("mf_230_1.0");
                }
                continue;
            }
            if (arg.startsWith(foldsPrefix)) {
                params.folds = new ArrayList<>();
                int min = 1;
                int max = 5;
                for (int i = min; i <= max; i++) {
                    if (arg.contains(String.valueOf(i))) {
                        params.folds.add(String.valueOf(i));
                    }
                }
                continue;
            }
            if (arg.startsWith(groupSizePrefix)) {
                params.groupSize = new ArrayList<>();
                if (arg.contains("all")) {
                    params.groupSize = Arrays.asList(2, 3, 4, 5, 6, 7, 8);
                }
                int min = 2;
                int max = 8;
                for(int i = min; i <= max; i++){
                    if (arg.contains(String.valueOf(i))) {
                        params.groupSize.add(i);
                    }
                }
                continue;
            }
            if (arg.startsWith(groupTypePrefix)) {
                params.groupTypes = new ArrayList<>();
                if (arg.contains("div")) {
                    params.groupTypes.add("div");
                }
                if (arg.contains("random")) {
                    params.groupTypes.add("random");
                }
    
                if (arg.contains("sim")) {
                    params.groupTypes.add("sim");
                }
                continue;
            }
        }
        // checks
        System.out.println(params.datasets);
        System.out.println(params.individualRecFileName);
        System.out.println(params.folds);
        System.out.println(params.groupSize);
        System.out.println(params.groupTypes);
        
        return params;
    }
}