import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by joak on 4/16/16.
 * seg plate from source pic
 */
class segPlate {

    private static final int GaussianBlurSize = 5;
    private static final int MorphSizeWidth = 17;
    private static final int MorphSizeHeight = 3;
    private static final int SobelScale = 1;
    private static final int SobelDelta = 0;
    private static final int SobelDDepth = CvType.CV_16S;
    private static final int SOBEL_X_WEIGHT = 1;
    private static final int M_ANGLE = 30;
    private static final int TARGET_WIDTH = 136;
    private static final int TARGET_HEIGHT = 36;
    private static final int TARGET_TYPE = CvType.CV_8UC3;
//    private static ImShow im = new ImShow("test");

    static void plateLocate(Mat in, List<Mat> out){
        List<RotatedRect> rects_plate_containing = new ArrayList<>();

        List<Rect> bound_rects = new ArrayList<>();
        sobelFirst(in, bound_rects);

//        for (int i = 0; i < bound_rects.size(); i++){
//            Mat bound_mat = in.submat(bound_rects.get(i));
//            Imgcodecs.imwrite(i + ".jpg", bound_mat);
//        }

        List<Rect> bound_rect_parts = new ArrayList<>();
        List<Rect> bound_rect_normal = new ArrayList<>();
        // extend unavailable area
        for (Rect roi : bound_rects){
            double f_ratio = roi.width * 1.0f / roi.height;
            if (f_ratio < 3.0 && f_ratio > 1.0 && roi.height < 120 ){
                // expand width
                roi.x = (int)(roi.x - roi.height * (4 - f_ratio));
                if (roi.x < 0) roi.x = 0;
                roi.width = (int)(roi.width + roi.height * 2 * (4 - f_ratio));
                if (roi.width + roi.x >= in.cols())
                    roi.width = in.cols() - roi.x;

                roi.y = (int)(roi.y - roi.height * 0.08f);
                roi.height = (int)(roi.height * 1.16f);

                bound_rect_parts.add(roi);
            } else bound_rect_normal.add(roi);
        }

        // secondary process for split part
//        System.out.println("bound_rect_parts:" + bound_rect_parts.size());
        for (Rect bound_part : bound_rect_parts){
            Point ref_point = new Point(bound_part.x, bound_part.y);

            double x = bound_part.x > 0 ? bound_part.x : 0;
            double y = bound_part.y > 0 ? bound_part.y : 0;

            double width = x + bound_part.width < in.cols() ?
                    bound_part.width : in.cols() - x;
            double height = y + bound_part.height < in.rows() ?
                    bound_part.height : in.rows() - y;

            Rect safe_bound_rect = new Rect((int)x, (int)y, (int)width, (int)height);
            Mat bound_mat = in.submat(safe_bound_rect);

            // secondary sobel search(part)
            sobelSecondPart(bound_mat, ref_point, rects_plate_containing);
        }

//        System.out.println("bound_rect_normal:" + bound_rect_normal.size());
        for (Rect bound_rect : bound_rect_normal){
            Point ref_point = new Point(bound_rect.x, bound_rect.y);

            double x = bound_rect.x > 0 ? bound_rect.x : 0;
            double y = bound_rect.y > 0 ? bound_rect.y : 0;

            double width = x + bound_rect.width < in.cols() ?
                    bound_rect.width : in.cols() - x;
            double height = y + bound_rect.height < in.rows() ?
                    bound_rect.height : in.rows() - 1;

            Rect safe_bound_rect = new Rect((int)x, (int)y, (int)width, (int)height);
            Mat bound_mat = in.submat(safe_bound_rect);

            // secondary sobel search
            sobelSecond(bound_mat, ref_point, rects_plate_containing);
        }

        levelPlate(in, rects_plate_containing, out);

    }

    private static void sobelFirst(Mat in, List<Rect> out){

        Mat img_mor = picOperate(in, GaussianBlurSize, MorphSizeWidth, MorphSizeHeight);

        // find contours of possibles plates
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(img_mor, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

//        Mat in_copy = in.clone();

        // iterate to each contour found
        List<RotatedRect> rects = new ArrayList<>();
        for (MatOfPoint itc:contours){
            MatOfPoint2f itc_2f = new MatOfPoint2f(itc.toArray());
            RotatedRect mr = Imgproc.minAreaRect(itc_2f);

//            if (mr.size.width > 230 && mr.size.height < 75){
//                System.out.println(mr.size);
//                Point[] rect_points = new Point[4];
//                mr.points(rect_points);
//                for (int j = 0; j < 4; j++){
//                    Imgproc.line(in_copy, rect_points[j], rect_points[(j + 1) % 4],
//                            new Scalar(0, 255, 255), 1);
//                }
//            }

            if (isRightSize(mr)){
                rects.add(mr);
            }
        }

//        im.showImage(in_copy);

        for (RotatedRect roi : rects){
            List<Rect> safeRect = new ArrayList<>();
            if (!calcSafeRect(roi, in, safeRect)) continue;
            out.addAll(safeRect);
        }

//        Mat in_copy = in.clone();
//        for (RotatedRect roi : rects){
//            Point[] rect_points = new Point[4];
//            roi.points(rect_points);
//            for (int j = 0; j < 4; j++){
//                Imgproc.line(in_copy, rect_points[j], rect_points[(j + 1) % 4],
//                        new Scalar(0, 255, 255), 1);
//            }
//        }
//        im.showImage(in_copy);
    }

    private static void sobelSecondPart(Mat in, Point ref_point, List<RotatedRect> out_rects){
        // refined operator
        Mat mat_threshold = picOperateRefined(in, 3, 6, 2);
        Mat temp_mat_threshold = mat_threshold.clone();

        utils.clearRivet(temp_mat_threshold);
        int[] pos = new int[2];
        // 0 left, 1 right
        if (findLeftRightBoundary(temp_mat_threshold, pos)){

            if (pos[0] != 0 && pos[1] != 0){
                // link two separated plate parts
                int pos_y = (int)(mat_threshold.rows() * 0.5);
                for (int i = pos[0] + (int)(mat_threshold.rows() * 0.1);
                        i < pos[1] - 4; i++)
                mat_threshold.put(pos_y, i, 255);
            }

            // discard useless part outside boundary
            for (int i = 0; i < mat_threshold.rows(); i++){
                mat_threshold.put(i, pos[0], 0);
                mat_threshold.put(i, pos[1], 0);
            }
        }

        // find contours of possibles plates
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mat_threshold, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        // iterate to each contour found
        for (MatOfPoint itc:contours){
            MatOfPoint2f itc_2f = new MatOfPoint2f(itc.toArray());
            RotatedRect mr = Imgproc.minAreaRect(itc_2f);
            if (isRightSize(mr)){
                Point ref_center = new Point(mr.center.x + ref_point.x ,
                        mr.center.y + ref_point.y);
                RotatedRect ref_roi = new RotatedRect(ref_center, mr.size, mr.angle);
                out_rects.add(ref_roi);
            }
        }

    }

    private static void sobelSecond(Mat in, Point ref_point, List<RotatedRect> out_rects){
        Mat mat_threshold = picOperate(in, 3, 10, 3);

        // find contours of possibles plates
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mat_threshold, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        // iterate to each contour found
        for (MatOfPoint itc:contours){
            MatOfPoint2f itc_2f = new MatOfPoint2f(itc.toArray());
            RotatedRect mr = Imgproc.minAreaRect(itc_2f);
            if (isRightSize(mr)){
                Point ref_center = new Point(mr.center.x + ref_point.x ,
                        mr.center.y + ref_point.y);
                RotatedRect ref_roi = new RotatedRect(ref_center, mr.size, mr.angle);
                out_rects.add(ref_roi);
            }
        }
    }

    private static Mat picOperate(Mat in, int blurSize, int morphW, int morphH){
        // convert image to gray
        Mat img_gray = new Mat();
        Imgproc.cvtColor(in, img_gray, Imgproc.COLOR_BGR2GRAY);

        // blur
        Mat img_blur = new Mat();
        Imgproc.GaussianBlur(img_gray, img_blur,
                new Size(blurSize, blurSize), 0, 0, Core.BORDER_DEFAULT);

        // x direction sobel
        Mat grad_x = new Mat();
        Mat abs_grad_x = new Mat();
        Imgproc.Sobel(img_blur, grad_x, SobelDDepth, 1, 0, 3, SobelScale, SobelDelta,
                Core.BORDER_DEFAULT);
        Core.convertScaleAbs(grad_x, abs_grad_x);

        // y direction sobel
        Mat grad_y = new Mat();
        Mat abs_grad_y = new Mat();
        Imgproc.Sobel(img_blur, grad_y, SobelDDepth, 0, 1, 3, SobelScale, SobelDelta,
                Core.BORDER_DEFAULT);
        Core.convertScaleAbs(grad_y, abs_grad_y);

        Mat img_grad = new Mat();
        Core.addWeighted(abs_grad_x, SOBEL_X_WEIGHT, abs_grad_y, 0, 0, img_grad);

        // threshold
        Mat img_threshold = new Mat();
        Imgproc.threshold(img_grad, img_threshold, 0, 255,
                Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);

        // morphologyEX
        Mat img_morphology = new Mat();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                new Size(morphW, morphH));
        Imgproc.morphologyEx(img_threshold, img_morphology, Imgproc.MORPH_CLOSE, element);

        return img_morphology.clone();
    }

    private static Mat picOperateRefined(Mat in, int blurSize, int morphW, int morphH){
        // convert image to gray
        Mat mat_gray = new Mat();
        Imgproc.cvtColor(in, mat_gray, Imgproc.COLOR_BGR2GRAY);

        // gaussian blur
        Mat mat_blur = new Mat();
        Imgproc.GaussianBlur(mat_gray, mat_blur, new Size(blurSize, blurSize),
                0, 0, Core.BORDER_DEFAULT);

        Mat grad_x = new Mat();
        Mat grad_y = new Mat();
        Mat abs_grad_x = new Mat();
        Mat abs_grad_y = new Mat();

        Imgproc.Sobel(mat_gray, grad_x, SobelDDepth, 1, 0, 3, SobelScale, SobelDelta,
                Core.BORDER_DEFAULT);
        Core.convertScaleAbs(grad_x, abs_grad_x);

        Imgproc.Sobel(mat_gray, grad_y, SobelDDepth, 0, 1, 3, SobelScale, SobelDelta,
                Core.BORDER_DEFAULT);
        Core.convertScaleAbs(grad_y, abs_grad_y);

        Mat grad = new Mat();
        Core.addWeighted(abs_grad_x, SOBEL_X_WEIGHT, abs_grad_y, 0, 0, grad);

        Mat mat_threshold = new Mat();
        Imgproc.threshold(grad, mat_threshold, 0, 255,
                Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                new Size(morphW, morphH));
        Mat mat_morph = new Mat();
        Imgproc.morphologyEx(mat_threshold, mat_morph, Imgproc.MORPH_CLOSE, element);

        return mat_morph;
    }

    private static boolean isRightSize(RotatedRect candidate){
        double error = 0.25;
        // china car plate size: 440X140
        double ratio = (double)440 / 140;
        // set a min and max area.
        double min = 44 * 14 * 2;
        double max = 44 * 14 * 35;
        // get only patches that match to a respect ratio
        double r_min = ratio - ratio * error;
        double r_max = ratio + ratio * error;

        double area = candidate.size.width * candidate.size.height;
        double r = candidate.size.width / candidate.size.height;
        if (r < 1){
            r = candidate.size.height / candidate.size.width;
        }

        return !((area < min || area > max) || (r < r_min || r > r_max));
    }

    private static boolean calcSafeRect(RotatedRect in, Mat img_src, List<Rect> safeRect){
        Rect boundRect = in.boundingRect();
        double tl_x = (boundRect.x > 0)? boundRect.x : 0;
        double tl_y = (boundRect.y > 0)? boundRect.y : 0;

        double br_x = (boundRect.x + boundRect.width < img_src.cols())?
                boundRect.x + boundRect.width - 1 : img_src.cols() - 1;
        double br_y = (boundRect.y + boundRect.height < img_src.rows())?
                boundRect.y + boundRect.height - 1 : img_src.rows() -1;

        double roi_width = br_x - tl_x;
        double roi_height = br_y - tl_y;

        if (roi_width <= 0 || roi_height <= 0 ){
            return false;
        } else {
            safeRect.add(new Rect((int)tl_x, (int)tl_y, (int)roi_width, (int)roi_height));
            return true;
        }
    }

    private static boolean findLeftRightBoundary(Mat in, int[] pos){
        // find boundary from two sides
        double span = in.rows() * 0.2f;

        // left side
        for (int i = 0; i < in.cols() - span - 1; i+=2){
            int whiteCount = 0;
            for (int k = 0; k < in.rows(); k++){
                for (int l = i; l < i + span; l++){
                    if (in.get(k, l)[0] == 255) whiteCount++;
                }
            }
            if (whiteCount * 1.0 / (span * in.rows()) > 0.36){
                pos[0] = i;
                break;
            }
        }

        // right side
        for (int i = in.cols() - 1; i > span; i-=2){
            int whiteCount = 0;
            for (int k = 0; k < in.rows(); k++){
                for (int l = i; l > i - span; l--){
                    if (in.get(k, l)[0] == 255) whiteCount++;
                }
            }
            if (whiteCount * 1.0 / (span * in.rows()) > 0.36){
                pos[1] = i;
                break;
            }
        }

        return pos[0] < pos[1];
    }

    private static boolean findLeftRightBoundaryStrict(Mat in, int[] pos){
        // find boundary from two sides
        // left side
        for (int i = 0; i < in.cols() * 0.25 ; i++){
            int whiteCount = 0;
            for (int j = 0; j < in.rows(); j++){
                if (in.get(j, i)[0] == 255) whiteCount++;
            }
            if (whiteCount * 1.0 / in.rows() > 0.9){
                pos[0] = i;
            }
        }

        pos[0]--;
        if (pos[0] < 0) pos[0] = 0;

        // right side
        for (int i = in.cols() - 1; i > in.cols() * 0.75; i--){
            int whiteCount = 0;
            for (int j = 0; j < in.rows(); j++){
                if (in.get(j, i)[0] == 255) whiteCount++;
            }
            if (whiteCount * 1.0 / in.rows() > 0.9){
                pos[1] = i;
            }
        }

        if (pos[1] + 1 < in.cols()) pos[1] ++;
        else pos[1] = in.cols() - 1;

        return pos[0] < pos[1];
    }

    private static void findTopBottomBoundary(Mat in, int[] pos){
        // 2 for top, 3 for bottom
        final int x = 7;
        for (int i = 0; i < in.rows() / 2; i++){
            int jumpCount = 0;
            int whiteCount = 0;
            for (int j = 0; j < in.cols() - 1; j++){
                if (in.get(i, j)[0] != in.get(i, j + 1)[0]) jumpCount++;
                if (in.get(i, j)[0] == 255) whiteCount++;
            }
            if ((jumpCount < x && whiteCount * 1.0 / in.cols() > 0.75)) pos[2] = i;
        }

        pos[2]--;
        if (pos[2] < 0) pos[2] = 0;

        for (int i = in.rows() - 1; i >= in.rows() / 2; i--){
            int jumpCount = 0;
            int whiteCount = 0;
            for (int j = 0; j < in.cols() - 1; j++){
                if (in.get(i, j)[0] != in.get(i, j + 1)[0]) jumpCount++;
                if (in.get(i, j)[0] == 255) whiteCount++;
            }
            if ((jumpCount < x && whiteCount * 1.0 / in.cols() > 0.75)) pos[3] = i;
        }

        pos[3]++;
        if (pos[3] >= in.rows()) pos[3] = in.rows() - 1;

        if (pos[2] >= pos[3]){
            pos[2] = 0;
            pos[3] = in.rows() - 1;
        }
    }

    private static void levelPlate(Mat in, List<RotatedRect> in_rects,
                                   List<Mat> out_plates){

        Mat in_b = picOperate(in, 3, 10, 3);

        for (RotatedRect rect : in_rects){
            double r_ratio = rect.size.width / rect.size.height;
            double r_angle = rect.angle;
            Size r_size = rect.size;
            if (r_ratio < 1) r_angle += 90;

//            Mat in_copy = in.clone();
//            Point[] rect_points = new Point[4];
//            rect.points(rect_points);
//            for (int j = 0; j < 4; j++){
//                Imgproc.line(in_copy, rect_points[j], rect_points[(j + 1) % 4],
//                        new Scalar(0, 255, 255), 1);
//            }
//            im.showImage(in_copy);

            // discard rotated rect whose angle greater than 30
            if (r_angle - M_ANGLE < 0 && r_angle + M_ANGLE > 0){
                List<Rect> safe_rect = new ArrayList<>();
                if (!calcSafeRect(rect, in, safe_rect)) continue;
                Rect safe_bound_rect = safe_rect.get(0);

                Mat bound_mat = in.submat(safe_bound_rect);
                Mat bound_mat_b = in_b.submat(safe_bound_rect);

                // conflict between submat and rotate
                r_size.width = safe_bound_rect.width;
                r_size.height = safe_bound_rect.height;
                Point ref_center = new Point(rect.center.x - safe_bound_rect.tl().x,
                        rect.center.y - safe_bound_rect.tl().y);
                Mat mat_deskew;
                // rotate and split
                List<Mat> mat_tmp = new ArrayList<>();
                Mat mat_rotated;
                Mat mat_rotated_b;
                if (rotateMat(bound_mat, mat_tmp, r_size, ref_center, r_angle))
                    mat_rotated = mat_tmp.get(0);
                else continue;

                mat_tmp.clear();
                if (rotateMat(bound_mat_b, mat_tmp, r_size, ref_center, r_angle))
                    mat_rotated_b = mat_tmp.get(0);
                else continue;

                double[] r_slope = new double[1];
                if (isDeflection(mat_rotated_b, r_angle, r_slope))
                    mat_deskew = affineMat(mat_rotated, r_slope[0]);
                else mat_deskew = mat_rotated;

                mat_deskew = purgeJam(mat_deskew);
                Mat mat_plate = new Mat(TARGET_HEIGHT, TARGET_WIDTH, TARGET_TYPE);
                if (mat_deskew.cols() * 1.0 / mat_deskew.rows() > 2.3 &&
                        mat_deskew.cols() * 1.0 / mat_deskew.rows() < 6){
                    //normalize mat resize
                    if (mat_deskew.cols() >= TARGET_WIDTH || mat_deskew.rows() >= TARGET_HEIGHT)
                        Imgproc.resize(mat_deskew, mat_plate, mat_plate.size(),
                                0, 0, Imgproc.INTER_AREA);
                    else
                        Imgproc.resize(mat_deskew, mat_plate, mat_plate.size(),
                                0, 0, Imgproc.INTER_CUBIC);

                    out_plates.add(mat_plate);
                }
            }
        }
    }

    private static Mat purgeJam(Mat in){
        Mat img_ff = floodfillPlate(in);
        Mat img_gray = new Mat();
        Imgproc.cvtColor(img_ff, img_gray, Imgproc.COLOR_BGR2GRAY);

        int width = img_ff.cols();
        int height = img_ff.rows();

        Mat img_threshold = new Mat();
        Rect tmp_rect = new Rect((int)(width * 0.15), (int)(height * 0.15),
                (int)(width * 0.7), (int)(height * 0.7));
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


        int[] pos = new int[4];
        // 0 for left, 1 for right
        // 2 for top, 3 for bottom
        pos[1] = img_threshold.cols() - 1;
        pos[3] = img_threshold.rows() - 1;
        findTopBottomBoundary(img_threshold, pos);
        if (findLeftRightBoundaryStrict(img_threshold, pos)){
            Rect a_rect = new Rect(pos[0], pos[2], pos[1] - pos[0], pos[3] - pos[2]);
            return img_ff.submat(a_rect);
        } else return in;
    }

    private static boolean rotateMat(Mat in, List<Mat> out, Size rect_size,
                                     Point center, double angle){
        Mat in_expand = new Mat((int)(in.rows() * 1.5), (int)(in.cols() * 1.5),
                in.type());
        double x = in_expand.cols() / 2 - center.x > 0 ?
                in_expand.cols() / 2 - center.x : 0;
        double y = in_expand.rows() / 2 - center.y > 0 ?
                in_expand.rows() / 2 - center.y : 0;
        double width = x + in.cols() < in_expand.cols() ?
                in.cols(): in_expand.cols() - x;
        double height = y + in.rows() < in_expand.rows() ?
                in.rows() : in_expand.rows() - y;

        if (width != in.cols() || height != in.rows()) return false;
        Mat mat_roi = in_expand.submat(new Rect((int)x, (int)y,
                (int)width, (int)height));
        Core.addWeighted(mat_roi, 0, in, 1, 0, mat_roi);

        Point center_new = new Point(in_expand.cols() * 0.5, in_expand.rows() * 0.5);

        Mat matrix_rot = Imgproc.getRotationMatrix2D(center_new, angle, 1);

        Mat mat_rotated = new Mat();
        Imgproc.warpAffine(in_expand, mat_rotated, matrix_rot,
                new Size(in_expand.cols(), in_expand.rows()), Imgproc.INTER_CUBIC);

        Mat img_crop = new Mat();
        Imgproc.getRectSubPix(mat_rotated, new Size(rect_size.width, rect_size.height),
                center_new, img_crop);

        out.add(img_crop);
        return true;
    }

    private static boolean isDeflection(Mat in, double angle, double[] slope){
        int[] comp_index = {in.rows() / 4, in.rows() / 2, in.rows() * 3 / 4};
        int[] len = new int[3];

        for (int i = 0; i < 3; i++){
            int index = comp_index[i];
            int j = 0;
            int pixel_value = 0;
            while (pixel_value == 0 && j < in.cols()) {
                pixel_value = (int)in.get(index, j++)[0];
            }
            len[i] = j;
        }

        double max_len = Math.max(len[2], len[0]);
        double min_len = Math.min(len[2], len[0]);

        double g = Math.tan(angle * Math.PI / 180);

        if (max_len - len[1] > in.cols() / 32 || len[1] - min_len > in.cols() / 32){
            // line gradient
            double slope_can_1 = (double)(len[2] - len[0]) / (double)comp_index[1];
            double slope_can_2 = (double)(len[1] - len[0]) / (double)comp_index[0];
            slope[0] = Math.abs(slope_can_1 - g) <= Math.abs(slope_can_2 - g) ?
                    slope_can_1 : slope_can_2;
            return true;
        } else {
            slope[0] = 0;
            return false;
        }
    }

    private static Mat affineMat(Mat in, double slope){
        Point[] dst_tri = new Point[3];
        Point[] pl_tri = new Point[3];

        double height = in.rows();
        double width = in.cols();
        double correct  = Math.abs(slope) * height;

        if (slope > 0){
            // deflect to right
            pl_tri[0] = new Point(0, 0);
            pl_tri[1] = new Point(width - correct - 1, 0);
            pl_tri[2] = new Point(correct, height - 1);
        } else {
            // deflect to left
            pl_tri[0] = new Point(correct, 0);
            pl_tri[1] = new Point(width - 1, 0);
            pl_tri[2] = new Point(0, height - 1);
        }

        dst_tri[0] = new Point(correct / 2, 0);
        dst_tri[1] = new Point(width - 1 - correct / 2, 0);
        dst_tri[2] = new Point(correct / 2, height - 1);

        MatOfPoint2f pl_tri_2f = new MatOfPoint2f(pl_tri);
        MatOfPoint2f dst_tri_2f = new MatOfPoint2f(dst_tri);
        Mat wrap_mat = Imgproc.getAffineTransform(pl_tri_2f, dst_tri_2f);

        Mat affine_mat = new Mat((int)height, (int)width, CvType.CV_8UC3);
        if (in.rows() > TARGET_HEIGHT || in.cols() > TARGET_WIDTH)
            Imgproc.warpAffine(in, affine_mat, wrap_mat, affine_mat.size(),
                    Imgproc.INTER_AREA);
        else
            Imgproc.warpAffine(in, affine_mat, wrap_mat, affine_mat.size(),
                    Imgproc.INTER_CUBIC);

        return affine_mat;
    }

    private static Mat floodfillPlate(Mat in){
        double min_size = (in.size().width < in.size().height)?
                in.size().width : in.size().height;
        min_size = min_size * 0.25;
        Mat mask = new Mat();
        mask.create(in.rows() + 2, in.cols() + 2, CvType.CV_8UC1);
        mask.setTo(new Scalar(0, 0,0));
        int lo_diff = 60;
        int up_diff = 60;
        int connectivity = 8;
        int new_mask_val = 255;
        int num_seeds = 8;
        Rect c_comp = new Rect();
        int flags = connectivity + (new_mask_val << 8) +
                Imgproc.FLOODFILL_MASK_ONLY + Imgproc.FLOODFILL_FIXED_RANGE;
        Random random = new Random();
        for (int i=0; i < num_seeds; i++){
            Point seed = new Point();
            seed.x = in.size().width * 0.35 + (double)random.nextInt() % (min_size);
            seed.y = in.size().height / 2 + (double)random.nextInt() % (min_size);
//            System.out.println("Seed:" + seed.x + "," + seed.y);
//            Imgproc.circle(in, seed, 1, new Scalar(0, 255, 255), -1);
            Imgproc.floodFill(in, mask, seed, new Scalar(255, 255, 255), c_comp,
                    new Scalar(lo_diff, lo_diff, lo_diff),
                    new Scalar(up_diff, up_diff, up_diff),
                    flags);
        }

        //Check new floodfill mask match for a correct patch.
        //Get all points detected for get Minimal rotated Rect
        List<Point> points_interest = new ArrayList<>();

        // iterate over Mat
        for (int i=0; i < mask.cols(); i++){
            for (int j=0; j < mask.rows(); j++){
                if (mask.get(j, i)[0] == 255) points_interest.add(new Point(i, j));
            }
        }

        MatOfPoint2f m2f_from_list = new MatOfPoint2f();
        m2f_from_list.fromList(points_interest);
        RotatedRect min_rect = Imgproc.minAreaRect(m2f_from_list);

        List<Rect> safeRect = new ArrayList<>();
        if (isRightSize(min_rect) && calcSafeRect(min_rect, in, safeRect)){
            return in.submat(safeRect.get(0));
        } else return in;
    }

}
