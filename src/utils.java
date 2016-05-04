import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by jo on 4/3/16.
 * utilities
 */
class utils {
    static int countNonZero(Mat in){
        int nonZero = 0;
        for (int i=0; i < in.rows(); i++){
            for (int j=0; j < in.cols(); j++){
                double[] pixel = in.get(i, j);
                if (pixel[0] != 0) nonZero ++;
            }
        }
        return nonZero;
    }

    private static int countBiggerValue(Mat in, int tValue){
        int iCount = 0;
        if (in.rows() > 1){
            for (int i = 0; i < in.rows(); i++){
                if (in.get(i, 0)[0] > tValue) iCount++;
            }
        } else {
            for (int i = 0; i < in.cols(); i++){
                if (in.get(0, i)[0] > tValue) iCount++;
            }
        }
        return iCount;
    }

    static void matPrint(Mat in){
        System.out.println();
        System.out.print(in.rows() + " * " + in.cols());
        for(int i = 0; i < in.rows(); i++){
            System.out.println();
            for(int j = 0; j < in.cols(); j++){
                System.out.print(in.get(i, j)[0] + " ");
            }
        }
        System.out.println();
    }

    static List<String> traverse(String path){
        List<String> out = new ArrayList<>();
        LinkedList<File> file_list = new LinkedList<>();
        file_list.add(new File(path));
        while (!file_list.isEmpty()){
            File file = file_list.removeFirst();
            File[] files = file.listFiles();
            if (files != null){
                for (File current_cursor: files){
                    if (current_cursor.isDirectory()){
                        file_list.add(current_cursor);
                    } else {
                        out.add(current_cursor.getAbsolutePath());
                    }
                }
            }
        }

        return out;
    }

    static void clearRivet(Mat in){
        final int x = 7;
        Mat jump = Mat.zeros(1, in.rows(), CvType.CV_32F);
        for (int i = 0; i < in.rows(); i++){
            int jumpCount = 0;
//            int whiteCount = 0;
            for (int j = 0; j < in.cols() - 1; j++){
                if (in.get(i, j)[0] != in.get(i, j + 1)[0]) jumpCount++;
//                if (in.get(i, j)[0] == 255) whiteCount++;
            }

            jump.put(0, i, (double)jumpCount);
        }

        for (int i = 0; i < in.rows(); i++){
            if (jump.get(0, i)[0] <= x){
                for (int j = 0; j < in.cols(); j++){
                    in.put(i, j, 0);
                }
            }
        }
    }

    private static Mat projectedHistogram(Mat in, FINALS.his_dir direction){
        int sz;
        if (direction == FINALS.his_dir.HORIZONTAL) sz = in.rows();
        else sz = in.cols();
        Mat mat_hist = Mat.zeros(1, sz, CvType.CV_32F);

        for (int i = 0; i < sz; i++){
            Mat data = (direction == FINALS.his_dir.HORIZONTAL) ?
                    in.row(i) : in.col(i);

            // count non-zero element number
            mat_hist.put(0, i, countBiggerValue(data, 20));
        }

        // normalize histogram
        Core.MinMaxLocResult mmlr = Core.minMaxLoc(mat_hist);
        if (mmlr.maxVal > 0) mat_hist.convertTo(mat_hist, -1, 1.0 / mmlr.maxVal, 0);

        return mat_hist;
    }

    static List<Double> extraFeatures(Mat in){

        Mat img_threshold = new Mat();
        Imgproc.threshold(in, img_threshold, 0, 255,
                Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);

        Mat v_hist = projectedHistogram(img_threshold, FINALS.his_dir.VERTICAL);
        Mat h_hist = projectedHistogram(img_threshold, FINALS.his_dir.HORIZONTAL);

        int num_cols = v_hist.cols() + h_hist.cols();
        List<Double> out = new ArrayList<>();

        // extract feature from histogram
        for (int i = 0; i < v_hist.cols(); i++){
            out.add(v_hist.get(0, i)[0]);
        }
        for (int i = 0; i < h_hist.cols(); i++){
            out.add(h_hist.get(0, i)[0]);
        }

        return out;
    }

    static List<Double> extraFeatures(Mat in, int size_data){
        // centering char
        Rect r = getCenterRect(in);
        Mat mat_dst = new Mat(in.rows(), in.cols(), CvType.CV_8UC1);
        mat_dst.setTo(new Scalar(0));
        Mat mat_src = in.submat(r);

        int span_x = (int)((in.width() - r.width) / 2.0f);
        int span_y = (int)((in.height() - r.height) / 2.0f);

        for (int i = 0; i < mat_src.rows(); i++){
            for (int j = 0; j < mat_src.cols(); j++){
                mat_dst.put(i + span_x, j + span_y,
                        mat_src.get(i, j)[0]);
            }
        }

        Mat low_res_data = new Mat();
        Imgproc.resize(mat_dst, low_res_data, new Size(size_data, size_data));

        // histogram features
        Mat v_hist = projectedHistogram(low_res_data, FINALS.his_dir.VERTICAL);
        Mat h_hist = projectedHistogram(low_res_data, FINALS.his_dir.HORIZONTAL);

        List<Double> features = new ArrayList<>();
        for (int j = 0; j < v_hist.cols(); j++){
            features.add(v_hist.get(0, j)[0]);
        }
        for (int j = 0; j < h_hist.cols(); j++){
            features.add(h_hist.get(0, j)[0]);
        }
        for (int i = 0; i < low_res_data.rows(); i++){
            for (int j = 0; j < low_res_data.cols(); j++){
                features.add(low_res_data.get(i, j)[0]);
            }
        }
        return features;
    }

    private static Rect getCenterRect(Mat in){
        int top = 0;
        int bottom = in.rows() - 1;

        // top and bottom
        for (int i = 0; i < in.rows(); i++){
            boolean find = false;
            for (int j = 0; j < in.cols(); j++){
                if (in.get(i, j)[0] > 20){
                    top = i;
                    find = true;
                    break;
                }
            }
            if (find) break;
        }

        for (int i = in.rows() - 1; i >= 0; i--){
            boolean find = false;
            for (int j = 0; j < in.cols(); j++){
                if (in.get(i, j)[0] > 20){
                    bottom = i;
                    find = true;
                    break;
                }
            }
            if (find) break;
        }

        int left = 0;
        int right = in.cols() - 1;
        // left and right
        for (int j = 0; j < in.cols(); j++){
            boolean find = false;
            for (int i = 0; i < in.rows(); i++){
                if (in.get(i, j)[0] > 20){
                    left = j;
                    find = true;
                    break;
                }
            }
            if (find) break;
        }

        for (int j = in.cols() - 1; j >= 0; j--){
            boolean find = false;
            for (int i = 0; i < in.rows(); i++){
                if (in.get(i, j)[0] > 20){
                    right = j;
                    find = true;
                    break;
                }
            }
            if (find) break;
        }

        return new Rect(left, top, right - left + 1, bottom - top + 1);
    }

}
