package gfar.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.javatuples.Pair;

public class LoadData {

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

    public static Map<Long, Pair<List<Long>, List<Double>>> loadGroupsWithUserPreferences(String filePath) {
        return loadGroupsWithUserPreferences(filePath, filePath + "_weights");
    }

    public static Map<Long, Pair<List<Long>, List<Double>>> loadGroupsWithUserPreferences(String filePath,
            String prefFilePath) {
        Scanner s = null;
        Scanner s_pref = null;
        try {
            s = new Scanner(new File(filePath));
            s_pref = new Scanner(new File(prefFilePath));
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

                if (id != id_pref) {
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

    public static Map<Long, Pair<List<Long>, List<Double>>> loadGroupsWithUniformUserPreferences(String filePath,
            Double uniformPreferenceValue) {
        Map<Long, List<Long>> originalWithoutPref = loadGroups(filePath);
        Map<Long, Pair<List<Long>, List<Double>>> newWithUniformPref = new HashMap<>();

        originalWithoutPref.forEach((key, list) -> {
            int size = list.size();
            List<Double> prefList = new ArrayList<Double>(Collections.nCopies(size, uniformPreferenceValue));
            newWithUniformPref.put(key, new Pair<>(list, prefList));
        });

        return newWithUniformPref;
    }

    public static Map<Long, Pair<List<Long>, List<Double>>> loadGroupsWithUserPreferencesFromMap(String filePath,
            Map<Long, List<Double>> groupUserPrefs) {
        Map<Long, List<Long>> originalWithoutPref = loadGroups(filePath);
        Map<Long, Pair<List<Long>, List<Double>>> newWithUniformPref = new HashMap<>();

        originalWithoutPref.forEach((key, list) -> {
            newWithUniformPref.put(key, new Pair<>(list, groupUserPrefs.get(key)));
        });

        return newWithUniformPref;
    }
}
