package gfar.util;

import java.util.List;

public class ListAnalyzer {

    public static double Eval(List<Double> list, String type) {
        double size = list.size();
        if (size != 0) {
            if (type.equals("MIN")) {
                return getMin(list);
            } else if (type.equals("MIN-MAX")) {
                return getMinMax(list);
            } else if (type.equals("AVG")) {
                return getAvg(list);
            } else if (type.equals("STD")) {
                return getStd(list);
            } else if(type.equals("ZERO")){
                return list.stream().mapToDouble(v -> v).filter(v -> v == 0.0).count() / size;
            }
        }
        return 0.0;
    }

    public static double getStd(List<Double> list) {
        double size = list.size();
        double sum = list.stream().mapToDouble(v -> v).sum();
        double mean = sum / size;
        double deviationSum = list.stream().mapToDouble(v -> Math.pow(v - mean, 2)).sum();
        double standardDeviation = Math.sqrt(deviationSum / (size - 1));
        return standardDeviation;
    }

    public static double getMin(List<Double> list) {
        return list.stream().mapToDouble(v -> v).min().orElse(0);
    }

    public static double getMinMax(List<Double> list) {
        double max = list.stream().mapToDouble(v -> v).max().orElse(0);
        if (max == 0)
            return 0.0;
        double min = list.stream().mapToDouble(v -> v).min().orElse(0);
        return min / max;
    }

    public static double getAvg(List<Double> list) {
        return list.stream().mapToDouble(v -> v).average().orElse(0);
    }

}
