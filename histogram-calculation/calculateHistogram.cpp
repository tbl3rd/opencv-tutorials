#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Create a new unobscured named window for image.
// Reset windows layout with when reset is not 0.
//
// The fudge works around how MacOSX lays out window decorations.
//
static void makeWindow(const char *window, const cv::Mat &image, int reset = 0)
{
    static int across = 1;
    static int moveCount = 0;
    if (reset) {
        across = reset;
        moveCount = 0;
    }
    const int overCount = moveCount % across;
    const int downCount = moveCount / across;
    const int moveX = overCount * image.cols;
    const int moveY = downCount * image.rows;
    const int fudge = downCount == 0 ? 0 : (1 + downCount);
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window, moveX, moveY + 23 * fudge);
    cv::imshow(window, image);
    ++moveCount;
}

// Return a histogram with binCount bins for the color values in plane.
//
static cv::Mat_<float> calculatePlane(const cv::Mat &plane, int binCount)
{
    static const int    imageCount     = 1;
    static const int    dimensionCount = 1;
    static const int    binCounts[]    = { binCount };
    static const float  histRange[]    = { 0, binCount };
    static const float *histRanges[]   = { histRange };
    static const bool   uniform        = true;
    static const bool   accumulate     = false;
    cv::Mat_<float>     result;
    cv::calcHist(&plane, imageCount, 0, cv::noArray(), result,
                 dimensionCount, binCounts, histRanges, uniform, accumulate);
    return result;
}

// Normalize the histogram in h for display in rows.
//
static void normalizeHistogram(cv::Mat &h, int rows)
{
    static const double alpha = 0;
    static const int normKind = cv::NORM_MINMAX;
    static const int dtype = -1;
    const double beta = rows;
    cv::normalize(h, h, alpha, beta, normKind, dtype, cv::noArray());
}

// Draw the normalized histogram in color on image.
//
static void drawHistogram(cv::Mat &image,
                          const cv::Mat_<float> &histogram,
                          const cv::Scalar &color)
{
    const int binWidth = cvRound(1.0 * image.cols / histogram.rows);
    cv::Point p0(0, image.rows - cvRound(histogram(0)));
    for (int i = 1; i < histogram.rows; ++i) {
        static const int thickness = 2;
        static const int lineType = 8;
        static const int shift = 0;
        const cv::Point p1(i * binWidth, image.rows - cvRound(histogram(i)));
        cv::line(image, p0, p1, color, thickness, lineType, shift);
        p0 = p1;
    }
}

// Return a new image with a histogram of colors in image.
//
static cv::Mat computeHistogram(const cv::Mat &image)
{
    static const cv::Scalar color[] = {
        cv::Scalar(255,   0,   0),      // BLUE
        cv::Scalar(  0, 255,   0),      // GREEN
        cv::Scalar(  0,   0, 255)       // RED
    };
    static const int colorCount = sizeof color / sizeof color[0];
    static const int binCount   = 256;
    cv::Mat plane[colorCount];
    cv::split(image, plane);
    cv::Mat_<float> hist[colorCount];
    for (int c = 0; c < colorCount; ++c) {
        hist[c] = calculatePlane(plane[c], binCount);
    }
    for (int c = 0; c < colorCount; ++c) {
        normalizeHistogram(hist[c], image.rows);
    }
    cv::Mat result = cv::Mat_<cv::Vec3b>::zeros(image.rows, image.cols);
    for (int c = 0; c < colorCount; ++c) {
        drawHistogram(result, hist[c], color[c]);
    }
    return result;
}

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            std::cout << av[0] << ": Press some key to quit." << std::endl;
            makeWindow("Source Image", image);
            const cv::Mat histogram = computeHistogram(image);
            makeWindow("Color Histogram", histogram);
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate histogram equalization."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
