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
    ++moveCount;
}

// Return a Hue histogram normalized [0, 255) for the hue-only image hue.
//
cv::Mat_<float> calculateHistogram(const cv::Mat &hue, int binCount)
{
    cv::Mat_<float> result;
    static const int max = std::numeric_limits<uchar>::max();
    static const cv::Mat noMask;
    static const float   hueRanges[]    = {0, max};
    static const float  *ranges[]       = {hueRanges};
    static const int     imageCount     = 1;
    static const int     dimensionCount = 1;
    static const bool    uniform        = true;
    static const bool    accumulate     = false;
    const int binCounts[] = {binCount};
    cv::calcHist(&hue, imageCount, 0, noMask, result,
                 dimensionCount, binCounts, ranges, uniform, accumulate);
    static const double  alpha          = 0.0;
    static const double  beta           = 1.0 * max;
    static const int     normKind       = cv::NORM_MINMAX;
    static const int     dtype          = -1;
    cv::normalize(result, result, alpha, beta, normKind, dtype, noMask);
    return result;
}

static cv::Mat calculateBackProjection(const cv::Mat &hue, const cv::Mat &hist)
{
    static const int max = std::numeric_limits<uchar>::max();
    static const int imageCount = 1;
    static const float hueRanges[] = {0, max};
    static const float *ranges[] = {hueRanges};
    static const double scale = 1.0;
    static const bool uniform = true;
    cv::Mat result;
    cv::calcBackProject(&hue, imageCount, 0, hist, result,
                        ranges, scale, uniform);
    return result;
}

static void drawHistogram(cv::Mat &image, const cv::Mat_<float> &hist)
{
    static const int max = std::numeric_limits<uchar>::max();
    static const cv::Scalar colorRed(0, 0, max);
    static const int lineType = -1;
    static const int shift = 0;
    const float scale = 1.0 * image.rows / max;
    const int binWidth = image.cols / hist.rows;
    image = cv::Mat::zeros(image.size(), CV_8UC3);
    for (int i = 0; i < hist.rows - 1; ++i) {
        const int height = image.rows - cvRound(scale * hist(i));
        const cv::Point lowerLeft(i * binWidth, image.rows);
        const cv::Point upperRight((i + 1) * binWidth, height);
        cv::rectangle(image, lowerLeft, upperRight, colorRed, lineType, shift);
    }
}


class BackProjectionDemo {

    const cv::Mat &bgrImage;
    cv::Mat hsvImage;
    cv::Mat hueOnly;
    cv::Mat histImage;
    cv::Mat backProjection;

    // The position of Hue Bins trackbar with binsBar < maxBins.
    //
    const int maxBins;
    int binsBar;

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnoredUseThisInstead,  void *p)
    {
        BackProjectionDemo *const pD = (BackProjectionDemo *)p;
        const int binCount = MAX(pD->binsBar, 1);
        assert(binCount < pD->maxBins);
        const cv::Mat_<float> hist = calculateHistogram(pD->hueOnly, binCount);
        pD->backProjection = calculateBackProjection(pD->hueOnly, hist);
        drawHistogram(pD->histImage, hist);
        cv::imshow("Histogram", pD->histImage);
        cv::imshow("Back Projection", pD->backProjection);
    }

    // Add a trackbar with label of range 0 to max in bar.
    //
    void makeTrackbar(const char *label, const char *window, int *bar, int max)
    {
        cv::createTrackbar(label, window, bar, max, &show, this);
    }

public:

    // Show this demo display window.
    //
    void operator()(void) { BackProjectionDemo::show(0, this); }

    // Construct a display with the caption c operating on source image s.
    //
    BackProjectionDemo(const cv::Mat &s):
        bgrImage(s), hsvImage(), hueOnly(),
        histImage(cv::Mat::zeros(s.size(), CV_8UC3)),
        backProjection(s), maxBins(256), binsBar(0)
    {
        static const int srcCount = 1;
        static const int dstCount = 1;
        static const int fromTo[] = {0, 0};
        static const int pairCount = sizeof fromTo / sizeof fromTo[0] / 2;
        cv::cvtColor(bgrImage, hsvImage, cv::COLOR_BGR2HSV);
        hueOnly.create(hsvImage.size(), hsvImage.depth());
        cv::mixChannels(&hsvImage, srcCount, &hueOnly, dstCount,
                        fromTo, pairCount);
        makeWindow("Original",        bgrImage, 3);
        makeWindow("HSV Image",       hsvImage);
        makeWindow("Hue Only",        hueOnly);
        makeWindow("Histogram",       histImage);
        makeWindow("Back Projection", backProjection);
        makeTrackbar("Hue Bins:", "Original", &binsBar, maxBins - 1);
        cv::imshow("Original",  bgrImage);
        cv::imshow("HSV Image", hsvImage);
        cv::imshow("Hue Only",  hueOnly);
        cv::imshow("Histogram", histImage);
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat bgr = cv::imread(av[1]);
        if (bgr.data) {
            std::cout << std::endl << "Press a key to quit." << std::endl;
            BackProjectionDemo bpDemo(bgr); bpDemo();
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate back projection."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image>" << std::endl
              << std::endl
              << "Where: <image> is an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/hand_sample2.jpg"
              << std::endl << std::endl;
    return 1;
}
