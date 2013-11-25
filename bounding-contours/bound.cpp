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


// A display to demonstrate bounding contours.
//
class DemoDisplay {

    const cv::Mat &srcImage;            // the original image
    cv::Mat drawing;                    // bounds in random colors

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

    // Draw bounds around contours with hierarchy over black drawing.
    //
    // The reference implies that drawContours() ignores hierarchy when
    // maxLevel is 0 though. =tbl
    //
    void drawBounds(const std::vector<std::vector<cv::Point> > &contour,
                    const std::vector<cv::Vec4i> &hierarchy)

    {
        static const cv::Mat black
            = cv::Mat::zeros(drawing.size(), drawing.type());
        static const bool orientClockwise = false;
        static const bool returnPoints = true;
        static const int polyThickness = 1;
        static const int boundThickness = 2 * polyThickness;
        static const int lineType = 8;
        static const int maxLevel = 0;
        static const double epsilon = 3.0;
        static const bool closed = true;
        static const cv::Point offset(0, 0);
        static const int shift = 0;
        const int size = contour.size();
        std::vector<std::vector<cv::Point> > polygon(size);
        std::vector<cv::Rect> rect(size);
        std::vector<cv::Point2f> center(size);
        std::vector<float> radius(size);
        for (int i = 0; i < size; ++i) {
            cv::approxPolyDP(contour[i], polygon[i], epsilon, closed);
            rect[i] = cv::boundingRect(polygon[i]);
            cv::minEnclosingCircle(polygon[i], center[i], radius[i]);
        }
        black.copyTo(drawing);
        for (int i = 0; i < size; ++i) {
            const cv::Scalar color = randomColor();
            cv::drawContours(drawing, polygon, i, color, polyThickness,
                             lineType, hierarchy, maxLevel, offset);
            cv::rectangle(drawing, rect[i].tl(), rect[i].br(),
                          color, boundThickness, lineType, shift);
            cv::circle(drawing, center[i], radius[i], color,
                       polyThickness, lineType, shift);
        }
    }

    // Find contours in srcImage, at threshold t less than max, and draw
    // bounding circles and rectangles in some random color on drawing.
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
        drawBounds(contour, hierarchy);
    }

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnoredUseThisInstead,  void *p)
    {
        DemoDisplay *const pD = (DemoDisplay *)p;
        assert(pD->bar <= pD->maxBar);
        const double value = pD->bar;
        pD->apply(value, pD->maxBar);
        cv::imshow("Bounds",  pD->drawing);
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

    // Demonstrate bounding polygonal contours with circles and rectangles.
    //
    DemoDisplay(const cv::Mat &s):
        srcImage(s), drawing(s.size(), CV_8UC3),
        bar(100), maxBar(std::numeric_limits<uchar>::max())
    {
        makeWindow("Original", srcImage, 2);
        makeWindow("Bounds",    drawing);
        makeTrackbar("Threshold:", "Original", &bar, maxBar);
        makeTrackbar("Threshold:", "Bounds",    &bar, maxBar);
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
    std::cerr << av[0] << ": Demonstrate bounding polygonal contours."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/jets.jpg"
              << std::endl << std::endl;
    return 1;
}
