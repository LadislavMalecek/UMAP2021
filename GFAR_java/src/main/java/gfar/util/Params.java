package gfar.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Params {
    public List<String> datasets = new ArrayList<>(Arrays.asList("ml1m", "kgrec"));
    public List<String> individualRecFileName = new ArrayList<>(Arrays.asList("mf_30_1.0", "mf_230_1.0"));

    public List<String> folds = new ArrayList<>(Arrays.asList("1", "2", "3", "4", "5"));
    public List<Integer> groupSize = new ArrayList<>(Arrays.asList(2, 3, 4, 8));
    public List<String> groupTypes = new ArrayList<>(Arrays.asList("div", "random", "sim"));
}