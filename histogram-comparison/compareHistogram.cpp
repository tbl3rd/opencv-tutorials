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

// Return an HSV encoding of the BGR image bgr.
//
static cv::Mat bgrToHsv(const cv::Mat bgr)
{
    cv::Mat result;
    cv::cvtColor(bgr, result, cv::COLOR_BGR2HSV);
    return result;
}

int main(int ac, const char *av[])
{
    static const int count = 3;
    cv::Mat bgr[count], hsv[1 + count];
    bool ok = ac == 1 + count;
    if (ok) {
        for (int i = 0; i < count; ++i) {
            bgr[i] = cv::imread(av[1 + i]);
            ok = ok && bgr[i].data;
        }
    }
    if (ok) for (int i = 0; i < count; ++i) hsv[i] = bgrToHsv(bgr[i]);
    if (ok) {
        const int rows = hsv[0].rows;
        const cv::Range lowRows(rows / 2, rows - 1);
        const cv::Range allCols(0, hsv[0].cols - 1);
        hsv[3] = hsv[0](lowRows, allCols);
    }
    cv::MatND hist[1 + count];
    static const int histCount = sizeof hist / sizeof hist[0];
    if (ok) {
        for (int i = 0; i < histCount; ++i) {
            static const cv::Mat noMask;
            static const int     hueBins        = 50;
            static const int     satBins        = 60;
            static const int     bins[]         = {hueBins, satBins};
            static const float   hueRanges[]    = {0, 256};
            static const float   satRanges[]    = {0, 180};
            static const float  *ranges[]       = {hueRanges, satRanges};
            static const int     imageCount     = 1;
            static const int     channels[]     = {0, 1};
            static const int     dimensionCount = 2;
            static const bool    uniform        = true;
            static const bool    accumulate     = false;
            cv::calcHist(&hsv[i], imageCount, channels, noMask, hist[i],
                         dimensionCount, bins, ranges, uniform, accumulate);
            static const double  alpha          = 0.0;
            static const int     normKind       = cv::NORM_MINMAX;
            static const int     dtype          = -1;
            static const double  beta           = 1.0;
            cv::normalize(hist[i], hist[i], alpha, beta,
                          normKind, dtype, noMask);
        }
    }
    if (ok) {
        for (int m = 0; m < 4; ++m) {
            double comparison[histCount] = {};
            for (int i = 0; i < histCount; ++i) {
                comparison[i] = cv::compareHist(hist[0], hist[i], m);
                std::cout << "comparison[" << i << "] == "
                          <<  comparison[i] << std::endl;
            }
            std::cout << "Method [" << m << "] "
                      << "Perfect, Base-Half, Base-Test(1), Base-Test(2) :";
            const char *separator = " "; 
            for (int i = 0; i < histCount; ++i) {
                std::cout << separator << comparison[i];
                separator = ", ";
            }
            std::cout << std::endl << std::endl;
        }
        std::cout << "Done." << std::endl;
    }
    if (ok) return 0;
    std::cerr << av[0] << ": Demonstrate histogram comparison."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <goal> <test0> <test1>" << std::endl
              << std::endl
              << "Where: <goal>, <test0>, and <test1> are color images."
              << std::endl
              << "       <goal> is the image to which <test0> and <test0>"
              << std::endl
              << "              are compared."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/hand*.jpg"
              << std::endl << std::endl;
    return 1;
}
