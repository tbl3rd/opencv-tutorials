#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>


static CvSVMParams makeSvmParams(int svm_type, int kernel_type,
                                 CvTermCriteria term_crit)
{
    CvSVMParams result;
    result.svm_type    = svm_type;
    result.kernel_type = kernel_type;
    result.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    return result;
}



int main(int, char *[])
{
    static const cv::Scalar black(  0,   0,   0);
    static const cv::Scalar white(255, 255, 255);
    static const cv::Scalar colors[4] = {black, white, white, white};
    float labels[4] = {1.0, -1.0, -1.0, -1.0};
    float trainingData[4][2] = {
        {501, 10}, {255, 10}, {501, 255}, {10, 501}
    };
    static const cv::Mat labelsMat(4, 1, CV_32FC1, labels);
    static const cv::Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    static const cv::Mat zeroIdx;
    static const cv::Mat varIdx = zeroIdx;
    static const cv::Mat sampleIdx = zeroIdx;
    static const cv::Vec3b green(  0, 255, 0);
    static const cv::Vec3b blue (255,   0, 0);
    const CvTermCriteria term_crit
        = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    const CvSVMParams params
        = makeSvmParams(CvSVM::C_SVC, CvSVM::LINEAR, term_crit);
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, varIdx, sampleIdx, params);
    cv::Mat image = cv::Mat::zeros(512, 512, CV_8UC3);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Mat sampleMat = (cv::Mat_<float>(1, 2) << i, j);
            const float response = SVM.predict(sampleMat);
            if (response == 1) {
                image.at<cv::Vec3b>(j, i)  = green;
            } else if (response == -1) {
                image.at<cv::Vec3b>(j, i)  = blue;
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        static const int radius = 5;
        static const int thickness = -1;
        static const int lineType = 8;
        const float *const v = trainingData[i];
        static const cv::Point center(v[0], v[1]);
        cv::circle(image, center, radius, colors[i], thickness, lineType);
    }
    const int count = SVM.get_support_vector_count();
    for (int i = 0; i < count; ++i) {
        static const cv::Scalar color(128, 128, 128);
        static const int thickness = 2;
        static const int lineType  = 8;
        const float *const v = SVM.get_support_vector(i);
        const cv::Point center(v[0], v[1]);
        static const int radius = 6;
        cv::circle(image, center, radius, color, thickness, lineType);
    }
    cv::imwrite("result.png", image);
    cv::imshow("SVM Simple Example", image);
    cv::waitKey(0);
}
