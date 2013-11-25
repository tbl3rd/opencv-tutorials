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


// A display to demonstrate finding convex hull over contours.
//
class DemoDisplay {

    const cv::Mat &srcImage;            // the original image
    cv::Mat itsThresholds;              // original masked by edges
    cv::Mat hullsImage;                 // contours in random colors

    int bar;                            // position of threshold trackbar
    const int maxBar;                   // maximum value of trackbar

    // Return a grayscale copy of image blurred by a kernel of size kSize.
    //
    static cv::Mat grayBlur(const cv::Mat &image, int kSize) {
        static const cv::Size kernel(kSize, kSize);
        cv::Mat gray, result;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        cv::blur(gray, result, kernel);
        return result;
    }

    // Return thresholds in image at threshold t less than max.
    //
    static const cv::Mat detectThresholds(const cv::Mat &image,
                                          double t, double max)
    {
        static const int kSize = 3;
        static const cv::Size kernel(kSize, kSize);
        static const cv::Mat grayBlur = DemoDisplay::grayBlur(image, kSize);
        cv::Mat result;
        cv::threshold(grayBlur, result, t, max, cv::THRESH_BINARY);
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

    // Draw contours in contour with hierarchy and convex hull around them
    // over black hullsImage.
    //
    // The reference implies that drawContours() ignores hierarchy when
    // maxLevel is 0 though. =tbl
    //
    void drawHulls(const std::vector<std::vector<cv::Point> > &contour,
                   const std::vector<cv::Vec4i> &hierarchy)

    {
        static const cv::Mat black
            = cv::Mat::zeros(hullsImage.size(), hullsImage.type());
        static const bool orientClockwise = false;
        static const bool returnPoints = true;
        static const int lineThickness = 1;
        static const int lineType = 8;
        static const int maxLevel = 0;
        static const cv::Point offset(0, 0);
        std::vector<std::vector<cv::Point> > hull(contour.size());
        black.copyTo(hullsImage);
        for (int i = 0; i < contour.size(); ++i) {
            const cv::Scalar color = randomColor();
            cv::convexHull(contour[i], hull[i], orientClockwise, returnPoints);
            cv::drawContours(hullsImage, contour, i, color, lineThickness,
                             lineType, hierarchy, maxLevel, offset);
            cv::drawContours(hullsImage, hull, i, color, lineThickness,
                             lineType, hierarchy, maxLevel, offset);
        }
    }

    // Find contours in srcImage, at threshold t less than max, and draw
    // their convex hull in some random color on hullsImage.
    //
    void apply(double t, double max)
    {
        static const int mode = cv::RETR_TREE;
        static const int method = cv::CHAIN_APPROX_SIMPLE;
        static const cv::Point offset(0, 0);
        std::vector<std::vector<cv::Point> > contour;
        std::vector<cv::Vec4i> hierarchy;
        const cv::Mat thresholds = detectThresholds(srcImage, t, max);
        cv::findContours(thresholds, contour, hierarchy, mode, method, offset);
        drawHulls(contour, hierarchy);
    }

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnoredUseThisInstead,  void *p)
    {
        DemoDisplay *const pD = (DemoDisplay *)p;
        assert(pD->bar <= pD->maxBar);
        const double value = pD->bar;
        pD->apply(value, pD->maxBar);
        cv::imshow("Hulls",  pD->hullsImage);
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

    // Demonstrate finding and drawing a convex hull on image s.
    //
    DemoDisplay(const cv::Mat &s):
        srcImage(s), hullsImage(s.size(), CV_8UC3),
        bar(100), maxBar(std::numeric_limits<uchar>::max())
    {
        makeWindow("Original", srcImage, 2);
        makeWindow("Hulls",    hullsImage);
        makeTrackbar("Threshold:", "Original", &bar, maxBar);
        makeTrackbar("Threshold:", "Hulls",    &bar, maxBar);
        cv::imshow("Original", srcImage);
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            DemoDisplay demo(image); demo();
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Find a convex hull around contours."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
