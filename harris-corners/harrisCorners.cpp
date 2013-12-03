#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Create a new unobscured named window for image.
// Reset windows layout with when reset is not 0.
//
// The 23 term works around how MacOSX decorates windows.
//
static void makeWindow(const char *window, const cv::Mat &image, int reset = 0)
{
    static int across = 1;
    static int count, moveX, moveY, maxY = 0;
    if (reset) {
        across = reset;
        count = moveX = moveY = maxY = 0;
    }
    if (count % across == 0) {
        moveY += maxY + 23;
        maxY = moveX = 0;
    }
    ++count;
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window, moveX, moveY);
    moveX += image.cols;
    maxY = std::max(maxY, image.rows);
}

// Return a grayscale copy of image.
//
static cv::Mat grayScale(const cv::Mat &image) {
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_RGB2GRAY);
    return result;
}

// Draw a black circle with center and radius on image.
//
static void drawCircle(cv::Mat &image, const cv::Point &center, int radius)
{
    static const cv::Scalar colorBlack(0, 0, 0);
    static const int thickness = 2;
    static const int lineKind = 8;
    static const int shift = 0;
    cv::circle(image, center, radius, colorBlack, thickness, lineKind, shift);
}

static cv::Mat normalizeImage(const cv::Mat &image)
{
    static const cv::Mat noMask;
    static const double alpha = 0.0;
    const double beta = 255.0;
    static const int normKind = cv::NORM_MINMAX;
    static const int dtype = CV_32FC1;
    cv::Mat result;
    cv::normalize(image, result, alpha, beta, normKind, dtype, noMask);
    return result;
}

// Return corners detected in image both normalized and scaled.
//
static void detectCorners(const cv::Mat &image,
                          cv::Mat &normalized, cv::Mat &scaled)
{
    static const int blockSize = 2;
    static const int aperture = 3;
    static const double k = 0.04;
    static const int borderKind = cv::BORDER_DEFAULT;
    cv::Mat corners = cv::Mat::zeros(image.size(), CV_32FC1);
    cv::cornerHarris(image, corners, blockSize, aperture, k, borderKind);
    normalized = normalizeImage(corners);
    cv::convertScaleAbs(normalized, scaled);
}


// A display to demonstrate finding corners using the Harris algorithm.
//
class DemoDisplay {

    const cv::Mat &sourceImage;
    const cv::Mat grayImage;
    cv::Mat normalizedCorners;
    cv::Mat scaledCorners;

    int bar;                            // position of threshold trackbar
    const int maxBar;                   // maximum value of trackbar

    const cv::Mat &apply(double threshold)
    {
        detectCorners(grayImage, normalizedCorners, scaledCorners);
        const cv::Mat_<float> &normalized = normalizedCorners;
        for (int j = 0; j < normalized.rows ; ++j) {
            for (int i = 0; i < normalized.cols; ++i) {
                const int value = normalized[j][i];
                if (value > threshold) {
                    static const int radius = 5;
                    const cv::Point center(i, j);
                    drawCircle(scaledCorners, center, radius);
                }
            }
        }
        return scaledCorners;
    }

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void showCorners(int positionIgnoredUseThisInstead, void *p)
    {
        DemoDisplay *const pD = (DemoDisplay *)p;
        assert (pD->bar <= pD->maxBar);
        const double threshold = pD->bar;
        const cv::Mat &scaled = pD->apply(threshold);
        cv::imshow("Corners", scaled);
    }

    // Add a trackbar with label of range 0 to max in bar.
    //
    void makeTrackbar(const char *label, const char *window, int *bar, int max)
    {
        cv::createTrackbar(label, window, bar, max, &showCorners, this);
    }

public:

    // Show this demo display window.
    //
    void operator()(void) { DemoDisplay::showCorners(0, this); }

    int threshold(void) { return bar; }

    // Find and display contours in image s.
    //
    DemoDisplay(const cv::Mat &s):
        sourceImage(s), grayImage(grayScale(s)),
        bar(200), maxBar(std::numeric_limits<uchar>::max())
    {
        grayImage.copyTo(normalizedCorners);
        grayImage.copyTo(scaledCorners);
        makeWindow("Source",  sourceImage, 2);
        makeWindow("Corners", scaledCorners);
        makeTrackbar("Threshold:", "Source",  &bar, maxBar);
        makeTrackbar("Threshold:", "Corners", &bar, maxBar);
        cv::imshow("Source", sourceImage);
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            std::cout << std::endl << av[0] << ": Press any key to quit."
                      << std::endl << std::endl;
            std::cout << av[0] << ": Useless below threshold 150."
                      << std::endl << std::endl;
            DemoDisplay demo(image); demo();
            std::cout << av[0] << ": Initial threshold is: "
                      << demo.threshold()<< std::endl << std::endl;
            cv::waitKey(0);
            std::cout << av[0] << ": Final threshold was: "
                      << demo.threshold() << std::endl << std::endl;
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate Harris corner finding."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> has an image with some corners in it."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/building.jpg"
              << std::endl << std::endl;
    return 1;
}
