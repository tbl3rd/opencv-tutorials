#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
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


class DemoDisplay {

    static const int kernelSize = 3;
    const cv::Mat &source;
    const cv::Mat gray;
    cv::Mat canny;
    cv::Mat output;

    int bar;
    const int maxBar;

    // Return a grayscale copy of image blurred by a kernel of size kSize.
    //
    static cv::Mat grayBlur(const cv::Mat &image, int kSize) {
        static const cv::Size kernel(kSize, kSize);
        cv::Mat gray, result;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        cv::blur(gray, result, kernel);
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

    // Report image moment calculations in contours and mu.
    //
    static void report(const std::vector<std::vector<cv::Point> > contours,
                       const std::vector<cv::Moments> &mu)
    {
        static const bool closed = true;
        static const int w = 15;
        std::cout << std::endl
                  << "Calculated Contour Arc Length and Areas" << std::endl
                  << std::endl;
        std::cout << "contour"
                  << std::setw(w) << "arcLength()"
                  << std::setw(w) << "mu.m00"
                  << std::setw(w) << "contourArea()"
                  << std::endl;
        for (int i = 0; i < contours.size(); ++i) {
            const std::vector<cv::Point> &contour = contours[i];
            const double area = cv::contourArea(contour);
            const double arc = cv::arcLength(contour, closed);
            std::cout << std::setw(7) << i
                      << std::fixed << std::setw(w) << std::setprecision(2)
                      << arc
                      << std::fixed << std::setw(w) << std::setprecision(1)
                      << mu[i].m00
                      << std::fixed << std::setw(w) << std::setprecision(1)
                      << area
                      << std::endl;
        }
    }

    // Draw a filled white circle of radius 4 at center on image.
    //
    static void drawWhiteCircle(cv::Mat &image, const cv::Point &center)
    {
        static const int radius = 4;
        static const cv::Scalar white(255, 255, 255);
        static const int fill = -1;
        static const int lineKind = 8;
        static const int shift = 0;
        cv::circle(image, center, radius, white, fill, lineKind, shift);
    }

    // Draw contours using hierarchy and mc on output.
    //
    void drawContours(const std::vector<std::vector<cv::Point> > contours,
                      const std::vector<cv::Vec4i> hierarchy,
                      const std::vector<cv::Point2f> &mc)
    {
        static const cv::Mat black = cv::Mat::zeros(canny.size(), CV_8UC3);
        static const int lineThickness = 2;
        static const int lineKind = 8;
        static const int maxLevel = 0;
        static const cv::Point offset(0, 0);
        black.copyTo(output);
        for (int i = 0; i< contours.size(); ++i) {
            const cv::Scalar color = randomColor();
            cv::drawContours(output, contours, i, color, lineThickness,
                             lineKind, hierarchy, maxLevel, offset);
            drawWhiteCircle(output, mc[i]);
        }
    }

    // Apply threshold to Canny edge detector and calculate the moments.
    // Draw the results on output.  And then report the calculations if
    // reportCalculations is true.
    //
    void apply(int threshold, bool reportCalculations)
    {
        static const bool binary = false;
        static const int ratio = 2;
        static const int mode = cv::RETR_TREE;
        static const int method = cv::CHAIN_APPROX_SIMPLE;
        static const cv::Point offset(0, 0);
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::Canny(gray, canny, threshold, ratio * threshold, kernelSize);
        cv::findContours(canny, contours, hierarchy, mode, method, offset);
        const int count = contours.size();
        std::vector<cv::Moments> mu(count);
        std::vector<cv::Point2f> mc(count);
        for (int i = 0; i < count; ++i) {
            mu[i] = cv::moments(contours[i], binary);
            mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
        }
        drawContours(contours, hierarchy, mc);
        if (reportCalculations) report(contours, mu);
    }

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnoredUseThisInstead,  void *p)
    {
        DemoDisplay *const pD = (DemoDisplay *)p;
        const double value = pD->bar;
        pD->apply(pD->bar, false);
        cv::imshow("Moments",  pD->output);
    }

    // Add a trackbar with label of range 0 to max in bar.
    //
    void makeTrackbar(const char *label, const char *window, int *bar, int max)
    {
        cv::createTrackbar(label, window, bar, max, &show, this);
    }

public:

    ~DemoDisplay() { apply(bar, true); }

    // Show this demo display.
    //
    void operator()(void) { DemoDisplay::show(0, this); }

    // Return the current threshold.
    //
    int threshold(void) { return bar; }

    // Demonstrate bounding polygonal contours with circles and rectangles.
    //
    DemoDisplay(const cv::Mat &s):
        source(s), gray(grayBlur(s, kernelSize)),
        bar(100), maxBar(std::numeric_limits<uchar>::max())
    {
        makeWindow("Original", source);
        makeWindow("Moments", source);
        makeTrackbar("Threshold:", "Original", &bar, maxBar);
        makeTrackbar("Threshold:", "Moments", &bar, maxBar);
        cv::imshow("Original", source);
    }
};

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            std::cout << std::endl << av[0] << ": Press a key to quit."
                      << std::endl << std::endl;
            DemoDisplay demo(image); demo();
            std::cout << "Initial threshold is: " << demo.threshold()
                      << std::endl;
            cv::waitKey(0);
            std::cout << "Final threshold was: " << demo.threshold()
                      << std::endl;
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate image moments."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/polygons.png"
              << std::endl << std::endl;
    return 1;
}
