import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.core.pmml.jaxbbindings.RadialBasisKernelType;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by jok on 5/1/16.
 * machine learning framework weka
 */
class predictWeka {

    private Classifier svm_cls;
    private Instances svm_attributes;
    private Classifier ann_en_cls;
    private Classifier ann_zh_cls;
    private Instances ann_attributes;

    predictWeka(){
        try {
            svm_cls = (Classifier) SerializationHelper.read("data/plate_svm.model");
            svm_attributes = ConverterUtils.DataSource.read("data/svm_attributes.arff");
            ann_en_cls = (Classifier) SerializationHelper.read("data/ann_en.model");
            ann_zh_cls = (Classifier) SerializationHelper.read("data/ann_zh.model");
            ann_attributes = ConverterUtils.DataSource.read("data/char_attributes.arff");
        } catch (Exception e){
            System.out.println("error with " + e);
        }
    }

    int predictSVM(Mat in){
        Mat img_gray = new Mat();
        Imgproc.cvtColor(in, img_gray, Imgproc.COLOR_BGR2GRAY);
        List<Double> features = utils.extraFeatures(img_gray);
        Instance instance = new DenseInstance(features.size());
        for (int i = 0; i < features.size(); i++){
            instance.setValue(i, features.get(i));
        }
        instance.setDataset(svm_attributes);
        svm_attributes.setClassIndex(svm_attributes.numAttributes() - 1);
        try {
            return (int)svm_cls.classifyInstance(instance);
        } catch (Exception e){
            System.out.println("error with: " + e);
        }
        return 0;
    }

    int predictANNEN(Mat in){
        List<Double> features = utils.extraFeatures(in, FINALS.CHAR_FEATURE_SIZE);
        Instance instance = new DenseInstance(features.size() + 1);
        for (int i = 0; i < features.size(); i++){
            instance.setValue(i, features.get(i));
        }
        // a misty bug that the instance needs 121 values not 120
        // but the 121th value is useless
        instance.setValue(features.size(), 0);
        instance.setDataset(ann_attributes);
        ann_attributes.setClassIndex(ann_attributes.numAttributes() - 1);
        try {
            return (int)ann_en_cls.classifyInstance(instance);
        } catch (Exception e){
            e.printStackTrace();
        }
        return 0;
    }

    int predictANNZH(Mat in){
        List<Double> features = utils.extraFeatures(in, FINALS.CHAR_FEATURE_SIZE);
        Instance instance = new DenseInstance(features.size() + 1);
        for (int i = 0; i < features.size(); i++){
            instance.setValue(i, features.get(i));
        }
        // a misty bug that the instance needs 121 values not 120
        // but the 121th value is useless
        instance.setValue(features.size(), 0);
        instance.setDataset(ann_attributes);
        ann_attributes.setClassIndex(ann_attributes.numAttributes() - 1);
        try {
            return (int)ann_zh_cls.classifyInstance(instance);
        } catch (Exception e){
            e.printStackTrace();
        }
        return 0;
    }

}
