#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>


// The values for the two classes of data in this example.
//
static const float greenStuff = 17.00;
static const float  blueStuff = 23.00;


// Train with LINEAR kernel Support Vector Classifier (C_SVC) with up to
// iterations times to achieve epsilon.
//
static cv::SVMParams makeSvmParams(void)
{
    static const int criteria = CV_TERMCRIT_ITER;
    static const int iterations = 1000 * 1000;
    static const double epsilon = std::numeric_limits<double>::epsilon();
    cv::SVMParams result;
    result.svm_type    = cv::SVM::C_SVC;
    result.kernel_type = cv::SVM::LINEAR;
    result.C           = 0.1;           // not sure what this adds
    result.term_crit   = cv::TermCriteria(criteria, iterations, epsilon);
    return result;
}

// Train svm on data and labels.
//
static void trainSvm(cv::SVM &svm, const cv::Mat &data, const cv::Mat labels)
{
    static const cv::Mat zeroIdx;
    static const cv::Mat varIdx = zeroIdx;
    static const cv::Mat sampleIdx = zeroIdx;
    static const cv::SVMParams params = makeSvmParams();
    svm.train(data, labels, varIdx, sampleIdx, params);
}

// Return count points of mostly (80%) separable training data randomly
// scattered in a rectangle of size.  The result is a float matrix of count
// rows and 2 columns where the X coordinates of the scattered points are
// in column 0 and the Y coordinates in column 1.
//
// The first separable (40%) points belong to the first region (x1), which
// labelData() will later give the value greenStuff.  The last separable
// (40%) points belong to another region (x2), which labelData() will later
// give the value blueStuff.  In between are 20% mixed between the two
// regions (xM), such that labelData() will give the first half of the
// mixed 20% the value greenStuff and the second half the value blueStuff.
//
// Consequently, the separable regions divide vertically into roughly equal
// areas somewhere along the X (column or width) axis and span the Y (row
// or height) axis.
//
// The draw*() routines will later color the regions by coloring the first
// half of the count points green and the second half blue.
//
static cv::Mat_<float> makeData(int count, const cv::Size &size)
{
    static const int uniform = cv::RNG::UNIFORM;
    static cv::RNG rng(666);
    const int cols = size.width;
    const int rows = size.height;
    cv::Mat_<float> result(count, 2, CV_32FC1);
    const cv::Mat aX = result.colRange(0, 1); // all X coordinates
    const cv::Mat aY = result.colRange(1, 2); // all Y coordinates
    const cv::Mat x1 = aX.rowRange(0.0 * count, 0.4 * count); // 40%
    const cv::Mat xM = aX.rowRange(0.4 * count, 0.6 * count); // 20%
    const cv::Mat x2 = aX.rowRange(0.6 * count, 1.0 * count); // 40%
    rng.fill(x1, uniform, 0.0 * cols, 0.4 * cols); //  40%
    rng.fill(xM, uniform, 0.4 * cols, 0.6 * cols); //  20%
    rng.fill(x2, uniform, 0.6 * cols, 1.0 * cols); //  40%
    rng.fill(aY, uniform, 0.0 * rows, 1.0 * rows); // 100%
    return result;
}

// Return half of data labeled greenStuff and half labeled blueStuff.
//
static cv::Mat_<float> labelData(const cv::Mat_<float> &data)
{
    const int rows = data.rows;
    const int half = rows / 2;
    cv::Mat_<float> result(rows, 1, CV_32FC1);
    result.rowRange(   0,  half).setTo(greenStuff);
    result.rowRange(half,  rows).setTo(blueStuff);
    return result;
}

// Draw on image the 2 classification regions predicted by svm.
//
static void drawRegions(cv::Mat_<cv::Vec3b> &image, const cv::SVM &svm)
{
    static const cv::Vec3b   pink(100, 100, 255);
    static const cv::Vec3b  green(  0, 100,   0);
    static const cv::Vec3b   blue(100,   0,   0);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            const cv::Mat sample = (cv::Mat_<float>(1,2) << i, j);
            const float response = svm.predict(sample);
            cv::Vec3b &pixel = image(j, i);
            if (response == greenStuff) {
                pixel = green;
            } else if (response == blueStuff) {
                pixel = blue;
            } else {
                pixel = pink;
                std::cerr << "Unexpected response from SVM::predict("
                          << sample << ") : " << response << std::endl;
            }
        }
    }
}

// Draw training data as count circles of radius 3 on image.
//
static void drawData(cv::Mat &image, const cv::Mat_<float> &data)
{
    static const cv::Scalar green(  0, 255,   0);
    static const cv::Scalar  blue(255,   0,   0);
    static const int radius = 3;
    static const int thickness = -1;
    static const int lineKind = 8;
    const int rows = data.rows;
    for (int i = 0; i < rows / 2; ++i) {
        const cv::Point center(data(i, 0), data(i, 1));
        cv::circle(image, center, radius, green, thickness, lineKind);
    }
    for (int i = rows / 2; i < rows; ++i) {
        const cv::Point center(data(i, 0), data(i, 1));
        cv::circle(image, center, radius, blue, thickness, lineKind);
    }
}

// Draw the support vectors in svm as circles of radius 6 in red.
//
static void drawSupportVectors(cv::Mat &image, const cv::SVM &svm)
{
    static const cv::Scalar red(0, 0, 255);
    static const int radius = 6;
    static const int thickness = 2;
    static const int lineKind = 8;
    const int count = svm.get_support_vector_count();
    std::cout << "support vector count == " << count << std::endl;
    for (int i = 0; i < count; ++i) {
        const float *const v = svm.get_support_vector(i);
        const cv::Point center(v[0], v[1]);
        std::cout << i << ": center == " << center << std::endl;
        cv::circle(image, center, radius, red, thickness, lineKind);
    }
}

int main(int, const char *[])
{
    static const char title[] = "SVM for Non-Linear Training Data";
    static const int count = 200;
    cv::Mat_<cv::Vec3b> image = cv::Mat::zeros(512, 512, CV_8UC3);
    const cv::Mat_<float> data = makeData(count, image.size());
    const cv::Mat_<float> labels = labelData(data);
    drawData(image, data);
    cv::imshow(title, image);
    std::cout << "Training SVM ... " << std::flush;
    cv::SVM svm;
    trainSvm(svm, data, labels);
    std::cout << "done." << std::endl;
    drawRegions(image, svm);
    drawData(image, data);
    drawSupportVectors(image, svm);
    cv::imshow(title, image);
    cv::imwrite("result.png", image);
    cv::waitKey(0);
}
