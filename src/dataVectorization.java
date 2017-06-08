

import java.io.*;
import java.util.*;


public class dataVectorization {

    // readData() reads files and returns its data as a string.
    private String readData(String filename) throws IOException {
        File file = new File(filename);
        String data;
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
            sb.append(line);
        }
        data = sb.toString();
        return data;
    }

    // removeSpecialChars() remove all unwanted characters like symbols, numbers etc.
    private String removeSpecialChars(String text) {
        return text.replaceAll("[^a-zA-Z ]", " ");
    }

    // getDataFrequency() counts word frequency from all Training and Testing email files.
    private Hashtable getDataFrequency() {
        String data = "null";
        Hashtable<String, Integer> hashtable = new Hashtable<String, Integer>();
        for (int i = 0; i <= 4326; i++) {
            try {
                String file = "E_Train/TRAIN_";
                file = file.concat(String.format("%05d", i));
                file = file.concat(".eml");
                data = removeSpecialChars(readData(file).toLowerCase());

                for (String Word : data.split("\\s+")) {
                    if (hashtable.containsKey(Word)) {
                        int n = hashtable.get(Word);
                        n++;
                        hashtable.put(Word, n);
                    } else {
                        hashtable.put(Word, 1);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
                break;
            }
        }
        for (int i = 0; i <= 4291; i++) {
            try {
                String file = "E_Test/TEST_";
                file = file.concat(String.format("%05d", i));
                file = file.concat(".eml");
                data = removeSpecialChars(readData(file).toLowerCase());

                for (String Word : data.split("\\s+")) {
                    if (hashtable.containsKey(Word)) {
                        int n = hashtable.get(Word);
                        n++;
                        hashtable.put(Word, n);
                    } else {
                        hashtable.put(Word, 1);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
                break;
            }
        }

        return hashtable;
    }

    // thresholdWordFrequency() removes all unwanted words from
    private HashMap thresholdWordFrequency(Hashtable<String, Integer> wordFrequency) {
        HashMap<String, Integer> thresholdWordFrequency = new HashMap<String, Integer>();
        Set<String> words = wordFrequency.keySet();
        for (String word : words) {
            if (wordFrequency.get(word) > 150 && word.length() > 2) {
                thresholdWordFrequency.put(word, wordFrequency.get(word));
            }
        }
        return thresholdWordFrequency;
    }

    // parameterMaker() gets all the parameter and write those to a txt file.
    private void parameterMaker(HashMap<String, Integer> thresholdWordFrequency) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter("parameterFD.txt", "UTF-8");
        for (Map.Entry<String, Integer> entry : thresholdWordFrequency.entrySet()) {
            System.out.println(entry.getKey() + " " + entry.getValue());
            writer.println(entry.getKey());
        }
        writer.close();
    }

    // parameterArray() reads parameterFD.txt and stores each parameter in an array
    private String[] parameterArray() throws IOException {
        BufferedReader in = new BufferedReader(new FileReader("parameterFD.txt"));
        String str;
        List<String> list = new ArrayList<String>();

        while ((str = in.readLine()) != null) {
            list.add(str);
        }

        String[] parameter = list.toArray(new String[0]);
        return parameter;
    }

    // linearStringSearch() is simple Linear Search
    private int linearStringSearch(String item, String[] data) {
        int pos;
        for (pos = 0; pos < data.length; pos++) {
            if (data[pos].equals(item)) {
                return pos;
            }
        }
        return -1;
    }

    // makeVector() converts a text email to vetor.
    private int[] makeVector(String data) throws IOException {
        String[] parameter = parameterArray();
        int[] dataVector = new int[parameter.length];
        data = removeSpecialChars(data.toLowerCase());
        for (String word : data.split("\\s+")) {
            int pos = linearStringSearch(word, parameter);
            if (pos >= 0) {
                dataVector[pos] = 1;
            }
        }
        return dataVector;
    }

    // getDataVector() returns Vectorized emails.
    public int[] getDataVector(String filename) throws IOException {
        String data = removeSpecialChars(readData(filename).toLowerCase());
        return makeVector(data);
    }

    public int[] getLabelArray() throws IOException {
        BufferedReader in = new BufferedReader(new FileReader("SPAMTrain.label"));
        Integer[] label;
        List<Integer> list = new ArrayList<Integer>();
        String str;
        while ((str = in.readLine()) != null){
            list.add(Integer.parseInt(str.split(" ")[0]));
        }
        label = list.toArray(new Integer[0]);
        return Arrays.stream(label).mapToInt(Integer::intValue).toArray();
    }
}