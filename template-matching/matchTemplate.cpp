#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Wait seconds or until some key is pressed.
// Return true if that key was 'q'.
// Otherwise return false.
//
static bool waitSeconds(int seconds)
{
    static const int oneSecondInMilliseconds = 1000;
    const int c = cv::waitKey(seconds * oneSecondInMilliseconds);
    return 'Q' == c || 'q' == c;
}

// Create a new unobscured named window for image.
// Reset windows layout with when reset is not 0.
//
// The 23 factor works around how MacOSX lays out window decorations.
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
        moveY += maxY + (1 + (count / across == 0)) * 23;
        maxY = moveX = 0;
    } 
    ++count;
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window, moveX, moveY);
    cv::imshow(window, image);
    moveX += image.cols;
    maxY = std::max(maxY, image.rows);
}

// The match methods available to cv::matchTemplate().
//
static const struct MatchMethod {
    bool useMin;                        // Use the minimum location if true.
    int kind;                           // Otherwise use the maximum.
    const char *name;
} matchMethod[] = {
    {true,  cv::TM_SQDIFF,        "cv::TM_SQDIFF"},
    {true,  cv::TM_SQDIFF_NORMED, "cv::TM_SQDIFF_NORMED"},
    {false, cv::TM_CCORR,         "cv::TM_CCORR"},
    {false, cv::TM_CCORR_NORMED,  "cv::TM_CCORR_NORMED"},
    {false, cv::TM_CCOEFF,        "cv::TM_CCOEFF"},
    {false, cv::TM_CCOEFF_NORMED, "cv::TM_CCOEFF_NORMED"}
};
static const int matchMethodCount = sizeof matchMethod / sizeof matchMethod[0];

// Show the results of matching tmp against src with method.
//
static void showMatch(const cv::Mat &src, const cv::Mat &tmp,
                      const MatchMethod &method)
{
    const cv::Size resultsSize = src.size() - tmp.size();
    cv::Mat results(resultsSize, CV_32FC1);
    cv::Mat display;
    src.copyTo(display);
    cv::matchTemplate(src, tmp, results, method.kind);
    static const cv::Mat noMask;
    static const double alpha = 0.0;
    static const double beta = 1.0;
    static const int dtype = -1;
    cv::normalize(results, results, alpha, beta,
                  cv::NORM_MINMAX, dtype, noMask);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(results, &minVal, &maxVal, &minLoc, &maxLoc, noMask);
    static const cv::Scalar colorBlack(0, 0, 0);
    static const int lineThickness = 2;
    static const int lineType = 8;
    static const int shift = 0;
    const cv::Point matchLoc = method.useMin ? minLoc : maxLoc;
    const cv::Point p(matchLoc.x + tmp.size().width,
                      matchLoc.y + tmp.size().height);
    cv::rectangle(display, matchLoc, p, colorBlack, lineThickness,
                  lineType, shift);
    cv::rectangle(results, matchLoc, p, colorBlack, lineThickness,
                  lineType, shift);
    makeWindow("Template Location", display);
    makeWindow(method.name, results);
}

// Demonstrate matches of tmp against src with all the available methods.
//
static bool showAllMatches(const cv::Mat &src, const cv::Mat &tmp)
{
    for (int i = 0; i < matchMethodCount; ++i) {
        cv::destroyAllWindows();
        makeWindow("Source Image", src, 4);
        makeWindow("Template Image", tmp);
        showMatch(src, tmp, matchMethod[i]);
        if (waitSeconds(0)) return true;
    }
    return false;
}

int main(int ac, const char *av[])
{
    if (ac == 3) {
        const cv::Mat src = cv::imread(av[1]);
        const cv::Mat tmp = cv::imread(av[2]);
        if (src.data && tmp.data) {
            std::cout << std::endl << "Press 'q' to quit." << std::endl;
            std::cout << std::endl << "Any other key to advance." << std::endl;
            showAllMatches(src, tmp);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate template matching."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image> <template>" << std::endl
              << std::endl
              << "Where: <image> is an image file."
              << std::endl
              << "       <template> is a small region of <image>."
              << std::endl << std::endl
              << "Example: " << av[0]
              << " ../resources/marilyn-jane.jpg ../resources/jane.jpg"
              << std::endl << std::endl;
    return 1;
}
