#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


static const int MAX_KERNEL_LENGTH = 31;

// Create a new unobscured named window for image.
// Reset windows layout with when reset is not 0.
//
// The 23 term works around how MacOSX decorates windows.
//
static void makeWindow(const char *window, const cv::Mat &image, int reset = 0)
{
    static int across = 3;
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

static bool displayDst(const cv::Mat &dst, const char *window, int delay)
{
    cv::imshow(window, dst);
    const int c = cv::waitKey(delay);
    if (c >= 0) return true;
    return false;
}
static bool displayShort(const cv::Mat &dst, const char *window)
{
    static const int DELAY_BLUR = 100;
    return displayDst(dst, window, DELAY_BLUR);
}
static bool displayLong(const cv::Mat &dst, const char *window)
{
    static const int DELAY_CAPTION = 1500;
    return displayDst(dst, window, DELAY_CAPTION);
}

static bool displayCaption(const cv::Mat &src, const char *caption)
{
    static const cv::Mat black = cv::Mat::zeros(src.size(), src.type());
    static const int face = cv::FONT_HERSHEY_COMPLEX;
    static const double scale = 1.0;
    static const cv::Scalar colorWhite(255, 255, 255);
    cv::Mat dst = black.clone();
    const cv::Point origin(dst.cols / 4, dst.rows / 2);
    makeWindow(caption, dst);
    cv::putText(dst, caption, origin, face, scale, colorWhite);
    return displayLong(dst, caption);
}

static bool showHomogeneousBlur(const cv::Mat &src)
{
    static const char caption[] = "Homogeneous Blur";
    static const cv::Point anchor(-1, -1);
    if (displayCaption(src, caption)) return true;
    for (int i = 1; i < MAX_KERNEL_LENGTH; i += 2) {
        const cv::Size kernelSize(i, i);
        cv::Mat dst;
        cv::blur(src, dst, kernelSize, anchor);
        if (displayShort(dst, caption)) return true;
    }
    return false;
}

static bool showGaussianBlur(const cv::Mat &src)
{
    static const char caption[] = "Gaussian Blur";
    static const double sigmaX = 0.0;
    static const double sigmaY = 0.0;
    if (displayCaption(src, caption)) return true;
    for (int i = 1; i < MAX_KERNEL_LENGTH; i += 2) {
        const cv::Size kernelSize(i, i);
        cv::Mat dst;
        cv::GaussianBlur(src, dst, kernelSize, sigmaX, sigmaY);
        if (displayShort(dst, caption)) return true;
    }
    return false;
}

static bool showMedianBlur(const cv::Mat &src)
{
    static const char caption[] = "Median Blur";
    if (displayCaption(src, caption)) return true;
    for (int i = 1; i < MAX_KERNEL_LENGTH; i += 2) {
        const int kernelSize = i;
        cv::Mat dst;
        cv::medianBlur(src, dst, kernelSize);
        if (displayShort(dst, caption)) return true;
    }
    return false;
}

static bool showBilateralBlur(const cv::Mat &src)
{
    static const char caption[] = "Bilateral Blur";
    if (displayCaption(src, caption)) return true;
    for (int i = 1; i < MAX_KERNEL_LENGTH; i += 2) {
        const int pixelNeighborhoodDiameter = i;
        const double sigmaColor = 2.0 * i;
        const double sigmaSpace = 0.5 * i;
        cv::Mat dst;
        cv::bilateralFilter(src, dst, pixelNeighborhoodDiameter,
                            sigmaColor, sigmaSpace);
        if (displayShort(dst, caption)) return true;
    }
    return false;
}

static bool showOriginal(const cv::Mat &src)
{
    static const char caption[] = "Original Image";
    if (displayCaption(src, caption)) return true;
    const cv::Mat dst = src.clone();
    if (displayLong(dst, caption)) return true;
    return false;
}


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat src = cv::imread(av[1], 1);
        if (!src.empty()) {
            const int stop
                =  showOriginal(src)
                || showHomogeneousBlur(src)
                || showGaussianBlur(src)
                || showMedianBlur(src)
                || showBilateralBlur(src)
                || displayCaption(src, "End: Press a key!");
            if (stop) return 0;
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate some blur filters."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
