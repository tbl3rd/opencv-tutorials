#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>


// Some colors.
//
static const cv::Scalar    sky(256, 256,   0);
static const cv::Scalar yellow(  0, 256, 256);
static const cv::Scalar    red(  0,   0, 255);
static const cv::Vec3b   green(  0, 255,   0);
static const cv::Vec3b    blue(255,   0,   0);

// Draw the count trainData on image in its respective colors.
//
static void drawTrainingData(cv::Mat &image, int count,
                             const float trainData[][2],
                             const cv::Scalar colors[])
{
    static const int radius = 5;
    static const int thickness = -1;
    static const int lineType = 8;
    for (int i = 0; i < count; ++i) {
        const float *const v = trainData[i];
        const cv::Point center(v[0], v[1]);
        cv::circle(image, center, radius, colors[i], thickness, lineType);
    }
}

// Draw in red circles on image the support vectors in svm.
//
static void drawSvm(cv::Mat &image, const cv::SVM &svm)
{
    static const int radius = 9;
    static const int thickness = 4;
    static const int lineType  = 8;
    const int count = svm.get_support_vector_count();
    std::cout << "support vector count == " << count << std::endl;
    for (int i = 0; i < count; ++i) {
        const float *const v = svm.get_support_vector(i);
        const cv::Point center(v[0], v[1]);
        std::cout << i << ": center == " << center << std::endl;
        cv::circle(image, center, radius, red, thickness, lineType);
    }
}

// Train with LINEAR kernel Support Vector Classifier (C_SVC) with up to
// 100 iterations to achieve an epsilon of 1e-6, whichever comes first.
//
// tbl: Turns out that only setting kernel_type to cv::SVM::LINEAR matters
// for this example.
//
static cv::SVMParams makeSvmParams(void)
{
    static const int criteria = CV_TERMCRIT_ITER;
    static const int iterationCount = 100;
    static const double epsilon = std::numeric_limits<double>::epsilon();
    cv::SVMParams result;
    result.svm_type    = cv::SVM::C_SVC;
    result.kernel_type = cv::SVM::LINEAR;
    result.term_crit   = cvTermCriteria(criteria, iterationCount, epsilon);
    return result;
}

// Train svm on the count values in trainData and labels, then draw on
// image the classifications associated with each label value in green
// (+1.0) or blue (-1.0) according to what the resulting svm predicts.
//
static void drawSvmRegions(cv::Mat_<cv::Vec3b> &image, cv::SVM &svm,
                           int count, float trainData[][2], float labels[])
{
    static const cv::Mat zeroIdx;
    static const cv::Mat varIdx = zeroIdx;
    static const cv::Mat sampleIdx = zeroIdx;
    static const cv::SVMParams params = makeSvmParams();
    const cv::Mat labelsMat(count, 1, CV_32FC1, labels);
    const cv::Mat trainDataMat(count, 2, CV_32FC1, trainData);
    svm.train(trainDataMat, labelsMat, varIdx, sampleIdx, params);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            const cv::Mat sampleMat = (cv::Mat_<float>(1, 2) << i, j);
            const float response = svm.predict(sampleMat);
            if (response == 1.0) {
                image(j, i) = green;
            } else if (response == -1.0) {
                image(j, i) = blue;
            } else {
                std::cerr << "Unexpected response from SVM::predict(): "
                          << response << std::endl;
                assert(!"expected response from SVM");
            }
        }
    }
}

// Show positive label values in black and negative values in white.
//
int main(int ac, const char *av[])
{
    const cv::Scalar colors[] = {      sky,    yellow,     yellow,   yellow};
    float labels[]            = {     +1.0,      -1.0,       -1.0,     -1.0};
    float trainData[][2]      = {{501, 10}, {255, 10}, {501, 255}, {10, 501}};
    cv::Mat_<cv::Vec3b> image = cv::Mat::zeros(512, 512, CV_8UC3);
    cv::SVM svm;
    std::cout << std::endl << av[0] << ": Press any key to quit." << std::endl;
    drawSvmRegions(image, svm, 4, trainData, labels);
    drawTrainingData(image, 4, trainData, colors);
    std::cout << "svm.get_var_count() == " << svm.get_var_count() << std::endl;
    drawSvm(image, svm);
    for (int i = 0; i < svm.get_var_count(); ++i)
    cv::imshow("SVM Simple Example", image);
    cv::waitKey(0);
}
