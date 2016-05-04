import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by jo on 3/30/16.
 * Main
 */

public class Main {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src_img = Imgcodecs.imread("pics/zheB7C289.jpg");
//        Mat src_img = Imgcodecs.imread("pics/yueA5Q951.jpg");
//        Mat src_img = Imgcodecs.imread("pics/zheA12210.jpg");

        predictWeka weka = new predictWeka();

        List<Mat> mat_candidates = new ArrayList<>();
        segPlate.plateLocate(src_img, mat_candidates);
        List<Mat> plates = new ArrayList<>();
        List<String> license = new ArrayList<>();
        boolean l_find = false;
        for (Mat p : mat_candidates){
            int predict = weka.predictSVM(p);
            if (predict == 0) {
                if (getLicence(p, license, weka)) {
                    l_find =true;
                    break;
                }
            }
        }
        if (l_find) System.out.println(license.toString());
        else System.out.println("Can't find license");

    }

    private static boolean getLicence(Mat in, List<String> license,
                                      predictWeka p){
        List<Mat> chars = new ArrayList<>();
        if (!segChars.contourCharacter(in, chars)) return false;
        for (int i = 0; i < chars.size(); i++){
            Mat c = chars.get(i);
            int predict;
            // chinese character
            if (i == 0){
                // predict accuracy 84%
                predict = p.predictANNZH(c);
                license.add(FINALS.str_chinese[predict]);
            } else  {
                // predict accuracy 98%
                predict = p.predictANNEN(c);
                license.add(FINALS.str_characters[predict]);
            }
        }
        return license.size() == 7;
    }

}
