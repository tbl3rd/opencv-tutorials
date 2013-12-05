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

// Return a random BGR color.
//
static cv::Scalar randomColor(void)
{
    static cv::RNG rng;
    const uchar red   = uchar(rng);
    const uchar green = uchar(rng);
    const uchar blue  = uchar(rng);
    return cv::Scalar(blue, green, red);
}

// Draw a circle with center and radius on image in a random color.
//
static void drawCircle(cv::Mat &image, const cv::Point &center, int radius)
{
    static const int thickness = -1;
    static const int lineKind = 8;
    static const int shift = 0;
    const cv::Scalar color = randomColor();
    cv::circle(image, center, radius, color, thickness, lineKind, shift);
}

// Return up to max corners in image.
//
static std::vector<cv::Point2f> findCorners(const cv::Mat image, int max)
{
    static const double quality = 0.01;
    static const double minDistance = 10.0;
    static const cv::Mat noMask;
    static const int blockSize = 3;
    static const bool useHarrisDetector = false;
    static const double k = 0.04;
    std::vector<cv::Point2f> result;
    cv::goodFeaturesToTrack(image, result, max, quality, minDistance,
                            noMask, blockSize, useHarrisDetector, k);
    return result;
}

class DemoDisplay {

    const cv::Mat &sourceImage;
    const cv::Mat grayImage;
    cv::Mat cornersImage;

    int bar;                            // position of max corners trackbar
    const int maxBar;                   // maximum value of trackbar

    // Return an image with corners marked to display.
    //
    const cv::Mat &apply()
    {
        static const int radius = 4;
        const int max = bar < 1 ? 1 : bar > maxBar ? maxBar : bar;
        const std::vector<cv::Point2f> corners = findCorners(grayImage, max);
        sourceImage.copyTo(cornersImage);
        for (int i = 0; i < corners.size(); ++i) {
            drawCircle(cornersImage, corners[i], radius);
        }
        return cornersImage;
    }

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void showCorners(int positionIgnoredUseThisInstead, void *p)
    {
        cv::imshow("Corners", ((DemoDisplay *)p)->apply());
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

    int maxCorners(void) { return bar; }

    // Find and display contours in image s.
    //
    DemoDisplay(const cv::Mat &s):
        sourceImage(s), grayImage(grayScale(s)), bar(23), maxBar(100)
    {
        sourceImage.copyTo(cornersImage);
        makeWindow("Corners", cornersImage);
        makeTrackbar("Max Corners:", "Corners", &bar, maxBar);
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            std::cout << std::endl << av[0] << ": Press any key to quit."
                      << std::endl << std::endl;
            DemoDisplay demo(image); demo();
            std::cout << av[0] << ": Initial maximum Corners is: "
                      << demo.maxCorners()<< std::endl << std::endl;
            cv::waitKey(0);
            std::cout << av[0] << ": Final maximum Corners was: "
                      << demo.maxCorners() << std::endl << std::endl;
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate Shi-Tomasi corner finding."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> has an image with some corners in it."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/building.jpg"
              << std::endl << std::endl;
    return 1;
}
