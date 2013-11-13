#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>


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

// Return a normalized Hue and Saturation histogram for the image hsv.
//
cv::Mat calculateHistogram(const cv::Mat &hsv)
{
    cv::Mat result;
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
    cv::calcHist(&hsv, imageCount, channels, noMask, result,
                 dimensionCount, bins, ranges, uniform, accumulate);
    static const double  alpha          = 0.0;
    static const int     normKind       = cv::NORM_MINMAX;
    static const int     dtype          = -1;
    static const double  beta           = 1.0;
    cv::normalize(result, result, alpha, beta, normKind, dtype, noMask);
    return result;
}


// Return a copy of the upper half of image.
//
static cv::Mat upperHalfOf(const cv::Mat &image)
{
    const int rows = image.rows;
    const cv::Range highRows(0, rows / 2);
    const cv::Range allCols(0, image.cols - 1);
    cv::Mat result = image(highRows, allCols);
    return result;
}

// Compare histogram[i] to histogram[0] for i = 0 to histogram.size().
// Find the corresponding image names at name[i].
//
static void compareHistograms(const std::vector<cv::Mat> &histogram,
                              const char *name[])
{
    static const struct Method {const char *name; int value;} method[] = {
        {"Correlation Match"     , CV_COMP_CORREL       },
        {"Intersection Match"    , CV_COMP_INTERSECT    },
        {"Chi-Square Distance"   , CV_COMP_CHISQR       },
        {"Bhattacharyya Distance", CV_COMP_BHATTACHARYYA}
    };
    static const int methodCount = sizeof method / sizeof method[0];
    std::cout << std::endl;
    std::cout << "Match means higher value is more similar." << std::endl;
    std::cout << "Distance means lower value is more similar." << std::endl;
    for (int m = 0; m < methodCount; ++m) {
        std::cout << std::endl << "Method: " << method[m].name << std::endl;
        double result[histogram.size()];
        for (int i = 0; i < histogram.size(); ++i) {
            const int mv = method[m].value;
            const double x = cv::compareHist(histogram[i], histogram[0], mv);
            std::cout << "        " << name[i] << " to " << name[0] << ": "
                      << x << std::endl;
        }
    }
    std::cout << std::endl << "Done." << std::endl;
}

// Display the count images in bgr and compute their HSV histograms.
// Then compare each histogram to the first one and report results.
//
static void showHistogramComparisons(int count,
                                     const char *name[],
                                     const cv::Mat bgr[])
{
    std::vector<cv::Mat> hsv(count);
    std::vector<cv::Mat> histogram(count);
    for (int i = 0; i < count; ++i) {
        makeWindow(name[i], bgr[i]);
        cv::cvtColor(bgr[i], hsv[i], cv::COLOR_BGR2HSV);
    }
    for (int i = 0; i < histogram.size(); ++i) {
        histogram[i] = calculateHistogram(hsv[i]);
    }
    compareHistograms(histogram, name);
    std::cout << "Press a key to quit." << std::endl;
    cv::waitKey(0);
}

// Read three images (named "Goal", "Tst0" and "Tst1") from the command
// line.  Copy the upper half of "Goal" to another image named "Half".
// Compute their HSV histograms, then compare them and show results.
//
int main(int ac, const char *av[])
{
    static const char *name[] = { "Goal", "Tst0", "Tst1", "Half" };
    static const int count = sizeof name / sizeof name[0];
    bool ok = ac == count;
    if (ok) {
        cv::Mat bgr[count];
        for (int i = 0; ok && i < count - 1; ++i) {
            bgr[i] = cv::imread(av[1 + i]);
            ok = ok && bgr[i].data;
        }
        if (ok) {
            bgr[count - 1] = upperHalfOf(bgr[0]);
            showHistogramComparisons(count, name, bgr);
            return 0;
        }
    }
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
