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


// A display to demonstrate finding contours over Canny edges.
//
class DemoDisplay {

protected:

    const cv::Mat &srcImage;            // the original image
    cv::Mat edgesImage;                 // the Canny edges
    cv::Mat cannyImage;                 // original masked by edges
    cv::Mat contoursImage;              // contours in random colors

    // Return a grayscale copy of image blurred by a kernel of size kSize.
    //
    static cv::Mat grayBlur(const cv::Mat &image, int kSize) {
        static const cv::Size kernel(kSize, kSize);
        cv::Mat gray, result;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        cv::blur(gray, result, kernel);
        return result;
    }

    // Apply Canny() with threshold to srcImage to construct an mask of
    // detected edges and overlay that mask back onto srcImage.
    //
    void cannyDetect(double threshold)
    {
        static const cv::Mat black
            = cv::Mat::zeros(cannyImage.size(), cannyImage.type());
        static const int ratio = 2;
        static const int kSize = 3;
        static const cv::Size kernel(kSize, kSize);
        static const cv::Mat grayBlur = DemoDisplay::grayBlur(srcImage, kSize);
        cv::Canny(grayBlur, edgesImage, threshold, ratio * threshold, kSize);
        black.copyTo(cannyImage);
        srcImage.copyTo(cannyImage, edgesImage);
    }

    // Return a random BGR color.
    //
    static cv::Scalar randomColor(void)
    {
        static const int max = std::numeric_limits<uchar>::max();
        static cv::RNG rng(12345);
        const int red   = rng.uniform(0, max);
        const int green = rng.uniform(0, max);
        const int blue  = rng.uniform(0, max);
        return cv::Scalar(blue, green, red);
    }

    // Find contours in edgesImage and draw each in some random color on
    // contoursImage.
    //
    // The reference implies that drawContours() ignores hierarchy when
    // maxLevel is 0 though. =tbl
    //
    void apply(double threshold)
    {
        static const cv::Mat black
            = cv::Mat::zeros(contoursImage.size(), contoursImage.type());
        static const int mode = cv::RETR_TREE;
        static const int method = cv::CHAIN_APPROX_SIMPLE;
        static const cv::Point offset(0, 0);
        static const int lineThickness = 2;
        static const int lineType = 8;
        static const int maxLevel = 0;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cannyDetect(threshold);
        cv::findContours(edgesImage, contours, hierarchy, mode, method, offset);
        black.copyTo(contoursImage);
        for (int i = 0; i< contours.size(); ++i) {
            const cv::Scalar color = randomColor();
            cv::drawContours(contoursImage, contours, i, color, lineThickness,
                             lineType, hierarchy, maxLevel, offset);
        }
    }

private:

    // The position of the Threshold trackbar and its maximum.
    //
    int thresholdBar;
    const int maxBar;

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnoredUseThisInstead,  void *p)
    {
        DemoDisplay *const pD = (DemoDisplay *)p;
        assert(pD->thresholdBar <= pD->maxBar);
        const double value = pD->thresholdBar;
        pD->apply(value);
        cv::imshow("Canny Edges", pD->edgesImage);
        cv::imshow("Canny Mask",  pD->cannyImage);
        cv::imshow("Contours",    pD->contoursImage);
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
    void operator()(void) { DemoDisplay::show(0, this); }

    // Construct a Canny demo display operating on source image s.
    //
    DemoDisplay(const cv::Mat &s):
        srcImage(s), thresholdBar(100), maxBar(255)
    {
        static const int max = std::numeric_limits<uchar>::max();
        assert(maxBar == max);
        srcImage.copyTo(cannyImage);
        contoursImage.create(cannyImage.size(), CV_8UC3);
        makeWindow("Canny Edges",      cannyImage);
        makeWindow("Canny Mask", cannyImage);
        makeWindow("Contours",   contoursImage);
        makeTrackbar("Threshold:", "Canny Edges", &thresholdBar,  maxBar);
        makeTrackbar("Threshold:", "Canny Mask",  &thresholdBar,  maxBar);
        makeTrackbar("Threshold:", "Contours",    &thresholdBar,  maxBar);
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            makeWindow("Original", image, 2);
            cv::imshow("Original", image);
            cv::createTrackbar("for alignment only", "Original", 0, 0, 0, 0);
            DemoDisplay demo(image); demo();
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate contour finding."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
