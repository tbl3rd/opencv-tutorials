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
    cv::imshow(window, image);
    moveX += image.cols;
    maxY = std::max(maxY, image.rows);
}


// A display to demonstrate thresholding in the Canny edge detector.
//
class DemoDisplay {

protected:

    // Source and destination images for apply().
    //
    const cv::Mat &srcImage;
    cv::Mat dstImage;

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
    void apply(double threshold)
    {
        static const int ratio = 3;
        static const int kSize = 3;
        static const cv::Size kernel(kSize, kSize);
        static const cv::Mat black
            = cv::Mat::zeros(srcImage.size(), srcImage.type());
        static const cv::Mat grayBlur
            = DemoDisplay::grayBlur(srcImage, kSize);
        cv::Mat edgeMask;
        cv::Canny(grayBlur, edgeMask, threshold, ratio * threshold, kSize);
        black.copyTo(dstImage);
        srcImage.copyTo(dstImage, edgeMask);
    }

private:

    // The window caption.
    //
    const char *const caption;

    // The position of the Threshold trackbar and its maximum.
    //
    int thresholdBar;
    const int maxThreshold;

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnoredUseThisInstead,  void *p)
    {
        DemoDisplay *const pD = (DemoDisplay *)p;
        const double value = pD->thresholdBar;
        pD->apply(value);
        cv::imshow(pD->caption, pD->dstImage);
    }

    // Add a trackbar in window with label of range 0 to max in bar.
    //
    void makeTrackbar(const char *label, const char *window, int *bar, int max)
    {
        cv::createTrackbar(label, window, bar, max, &show, this);
    }

public:

    // Show this demo display window.
    //
    void operator()(void) { DemoDisplay::show(0, this); }

    int threshold(void) { return thresholdBar; }

    // Construct a Canny demo display operating on source image s.
    //
    DemoDisplay(const cv::Mat &s):
        srcImage(s), caption("Edge Map"), thresholdBar(0), maxThreshold(100)
    {
        srcImage.copyTo(dstImage);
        makeWindow("Original", srcImage, 2);
        makeWindow(caption, dstImage);
        makeTrackbar("Threshold:", caption, &thresholdBar,  maxThreshold);
        makeTrackbar("Threshold:", "Original", &thresholdBar,  maxThreshold);
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            DemoDisplay demo(image); demo();
            std::cout << "Initial threshold is: " << demo.threshold()
                      << std::endl;
            cv::waitKey(0);
            std::cout << "Final threshold was: " << demo.threshold()
                      << std::endl;
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate Canny edge detection."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
