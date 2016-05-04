import com.csvreader.CsvWriter;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by jok on 5/1/16.
 * extract features from mat for machine learning
 */
class getTrainingData {

    static void plateSVM(){
        String path_to_plate = "pics/plates/";
        List<String> plate_img = utils.traverse(path_to_plate);
        String path_to_no_plate = "pics/no_plate/";
        List<String> no_plate_img = utils.traverse(path_to_no_plate);

        CsvWriter csv_plate = new CsvWriter("data/svm_plate.csv",
                ',', Charset.forName("UTF-8"));
        {
            // add csv file header
            List<String> tmp = new ArrayList<>();
            // 172 = 136 + 36;
            for (int i = 0; i < 172; i++) {
                tmp.add("arg" + i);
            }
            tmp.add("out");
            String[] content = tmp.toArray(new String[0]);
            try {
                csv_plate.writeRecord(content);
                csv_plate.flush();
            } catch (IOException e) {
                System.out.println("csv file write error");
            }
        }

        // iterate over plates folder
        for (String file: plate_img){
            System.out.println("deal with: " + file);
            Mat src_img = Imgcodecs.imread(file, 0);
            List<Double> features = utils.extraFeatures(src_img);
            writeCSV(features, csv_plate, "y");
        }

        // iterate over no plates folder
        for (String file: no_plate_img){
            System.out.println("deal with: " + file);
            Mat src_img = Imgcodecs.imread(file, 0);
            List<Double> features = utils.extraFeatures(src_img);
            writeCSV(features, csv_plate, "n");
        }

    }

    static void charANN(){
        String base_path = "pics/char_samples/";
        int num_non_zh = FINALS.str_characters.length;
        int num_zh = FINALS.str_chinese.length;

        // all characters
        CsvWriter csv_ann = new CsvWriter("data/ann.csv",
                ',', Charset.forName("UTF-8"));
        {
            // add csv file header
            List<String> tmp = new ArrayList<>();
            // 120 = 10 + 10 + 100;
            for (int i = 0; i < 120; i++) tmp.add("arg" + i);
            tmp.add("out");
            String[] content = tmp.toArray(new String[0]);
            try {
                csv_ann.writeRecord(content);
                csv_ann.flush();
            } catch (IOException e) {
                System.out.println("csv file write error");
            }
        }
        // ENGLISH
        CsvWriter csv_ann_eng = new CsvWriter("data/ann_eng.csv",
                ',', Charset.forName("UTF-8"));
        {
            // add csv file header
            List<String> tmp = new ArrayList<>();
            // 120 = 10 + 10 + 100;
            for (int i = 0; i < 120; i++) tmp.add("arg" + i);
            tmp.add("out");
            String[] content = tmp.toArray(new String[0]);
            try {
                csv_ann_eng.writeRecord(content);
                csv_ann_eng.flush();
            } catch (IOException e) {
                System.out.println("csv file write error");
            }
        }
        // CHINESE
        CsvWriter csv_ann_zh = new CsvWriter("data/ann_zh.csv",
                ',', Charset.forName("UTF-8"));
        {
            // add csv file header
            List<String> tmp = new ArrayList<>();
            // 120 = 10 + 10 + 100;
            for (int i = 0; i < 120; i++) tmp.add("arg" + i);
            tmp.add("out");
            String[] content = tmp.toArray(new String[0]);
            try {
                csv_ann_zh.writeRecord(content);
                csv_ann_zh.flush();
            } catch (IOException e) {
                System.out.println("csv file write error");
            }
        }

        // non-chinese characters
        int ann = 0;
        for (int i = 0; i < num_non_zh; i++){
            String t_directory = base_path + FINALS.str_characters[i];
            List<String> files = utils.traverse(t_directory);
            System.out.println("Deal with:" + t_directory);

            for (String t_file: files){
                Mat src_img = Imgcodecs.imread(t_file, 0);
                List<Double> features = utils.extraFeatures(src_img,
                        FINALS.CHAR_FEATURE_SIZE);
                writeCSV(features, csv_ann_eng, String.valueOf(i));
                writeCSV(features, csv_ann, String.valueOf(ann));
            }
            ann++;
        }

        // chinese characters
        for (int i = 0; i < num_zh; i++){
            String t_directory = base_path + FINALS.str_chinese[i];
            List<String> files = utils.traverse(t_directory);
            System.out.println("Deal with:" + t_directory);

            for (String t_file: files){
                Mat src_img = Imgcodecs.imread(t_file, 0);
                List<Double> features = utils.extraFeatures(src_img,
                        FINALS.CHAR_FEATURE_SIZE);
                writeCSV(features, csv_ann_zh, String.valueOf(i));
                writeCSV(features, csv_ann, String.valueOf(ann));
            }
            ann++;
        }
    }

    private static void writeCSV(List<Double> features, CsvWriter writer, String value){

        List<String> tmp = new ArrayList<>();
        features.forEach(x -> tmp.add(String.valueOf(x)));
        tmp.add(value);
        String[] content = tmp.toArray(new String[0]);
        try {
            writer.writeRecord(content);
            writer.flush();
        } catch (IOException e) {
            System.out.println("csv file write error");
        }
    }
}
