import org.opencv.imgproc.*;
import org.opencv.core.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jok on 5/2/16.
 * seg chars from plate
 */
class segChars {
    private static boolean isRightCharSize(Rect in){
        // char size 92x45
        double aspect = 45.0f / 90.0f;
        double in_aspect = (double)in.width / (double)in.height;
        double error = 0.6;
        double min_height = 15;
        double max_height = 45;
        // special aspect ratio for number 1
        double min_aspect = 0.05f;
        double max_aspect = aspect + aspect * error;

        return (in_aspect > min_aspect && in_aspect < max_aspect &&
        in.height >= min_height && in.height < max_height);
    }

    private static int findSpecialCharacter(List<Rect> in){
        double maxHeight = 0;
        double maxWidth = 0;

        for (Rect it_mr: in) {
            if (it_mr.size().height > maxHeight) {
                maxHeight = it_mr.size().height;
            }
            if (it_mr.size().width > maxWidth) {
                maxWidth = it_mr.size().width;
            }
        }

        int specIndex = 0;
        for (int i=0; i < in.size(); i++) {
            Rect it_mr = in.get(i);
            int mid_x = it_mr.x + it_mr.width / 2;
            if ((it_mr.width > maxWidth * 0.8 || it_mr.height > maxHeight * 0.8)
                    && (mid_x > 20 && mid_x < 40)) {
                specIndex = i;
            }
        }

        return specIndex;
    }

    private static boolean checkChineseChar(Rect zh_old, Rect zh_new){
        if (zh_old.x > 136 * 3 / 14) return false;
        int area_new = zh_new.width * zh_new.height;
        int area = zh_old.width * zh_old.height;
        double ratio = (double)area_new / (double)area;
        return (Math.abs(1.0 - ratio) <= 0.15)
                && (Math.abs(zh_old.x - zh_new.x) <= 10);
    }

    static boolean contourCharacter(Mat in, List<Mat> char_mats){
        // gray
        Mat img_gray = new Mat();
        Imgproc.cvtColor(in, img_gray, Imgproc.COLOR_BGR2GRAY);

        // threshold
        int width = img_gray.width();
        int height = img_gray.height();
        Mat img_threshold = new Mat();
        Rect tmp_rect = new Rect((int)(width * 0.1), (int)(height * 0.1),
             (int)(width * 0.8), (int)(height * 0.8));
        Mat tmp_mat = img_gray.submat(tmp_rect);
        double threshold_value = Imgproc.threshold(tmp_mat, new Mat(), 0, 255,
             Imgproc.THRESH_OTSU);
        Imgproc.threshold(img_gray, img_threshold, threshold_value, 255,
             Imgproc.THRESH_BINARY);
        // if it's yellow plate, invert binary result
        int whiteCount = 0;
        int blackCount = 0;
        for (int i = 0; i < img_threshold.rows(); i++){
         for (int j = 0; j < img_threshold.cols(); j++){
             if (img_threshold.get(i, j)[0] == 255) whiteCount++;
             else blackCount++;
         }
        }
        if (whiteCount > blackCount){
         Imgproc.threshold(img_gray, img_threshold, threshold_value, 255,
                 Imgproc.THRESH_BINARY_INV);
        }

        // clear rivet
        utils.clearRivet(img_threshold);
        Mat img_aux = img_threshold.clone();

        // find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(img_threshold, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        // draw contours
        List<Rect> available_mrs = new ArrayList<>();
        for (MatOfPoint itc: contours){
            Rect mr = Imgproc.boundingRect(itc);
            if (isRightCharSize(mr)){
                available_mrs.add(mr);
//                Imgproc.rectangle(in, mr.br(), mr.tl(), new Scalar(0, 255, 0), 2);
            }
        }

        // sort rects
        List<Rect> char_rects = new ArrayList<>();
        if (available_mrs.size() > 5) {
            available_mrs.sort((o1, o2)-> o1.x - o2.x);
            // check if the chinese character is found
            int specialIndex = findSpecialCharacter(available_mrs);
            Rect chinese;
            boolean useNewZhChar;
            if (specialIndex < available_mrs.size() && specialIndex > 0){
                chinese = available_mrs.get(specialIndex).clone();
                chinese.size().width *= 1.15f;
                int x_new = chinese.x - (int)(chinese.size().width * 1.15f);
                x_new = x_new > 0 ? x_new : 0;
                chinese.x = x_new;

                // check the original chinese char candidate is correct
                Rect r = available_mrs.get(specialIndex - 1);
                useNewZhChar = !checkChineseChar(r, chinese);
            } else return false;

            if (useNewZhChar){
                char_rects.add(chinese);
                if (specialIndex + 6 <= available_mrs.size())
                    char_rects.addAll(available_mrs.subList(specialIndex, specialIndex + 6));
                else return false;
            } else {
                char_rects.add(available_mrs.get(specialIndex - 1));
                if (specialIndex + 6 <= available_mrs.size())
                    char_rects.addAll(available_mrs.subList(specialIndex, specialIndex + 6));
                else return false;
            }
        }

        if (char_rects.size() != 7) return false;
        for (Rect r : char_rects){
            Mat aux_roi = new Mat(img_aux, r);
            // normalize size
            int h = aux_roi.rows();
            int w = aux_roi.cols();
            int char_size = FINALS.CHAR_SIZE;
            Mat mat_transform = Mat.eye(2, 3, CvType.CV_32F);
            int m = (w > h) ? w : h;
            mat_transform.put(0, 2, (m / 2 - w / 2));
            mat_transform.put(1, 2, (m / 2 - h / 2));

            Mat img_warp = new Mat(m, m, aux_roi.type());
            Imgproc.warpAffine(aux_roi, img_warp, mat_transform,
                    img_warp.size(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT,
                    new Scalar(0));

            Mat out = new Mat();
            Imgproc.resize(img_warp, out, new Size(char_size, char_size));
            char_mats.add(out);
        }

        return true;
    }
}
