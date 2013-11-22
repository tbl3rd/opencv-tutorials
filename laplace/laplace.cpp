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

// Return src after applying a default Gaussian blur with a kernel of size
// (kernelSize x kernelSize) and converting to grayscale and showing the
// results.
//
static cv::Mat showOriginalBlurGray(const cv::Mat &src, int kernelSize)
{
    static const double sigmaX = 0.0;
    static const double sigmaY = 0.0;
    static const int borderKind = cv::BORDER_DEFAULT;
    const cv::Size kernel(kernelSize, kernelSize);
    makeWindow("Original", src, 2);
    cv::Mat blur;
    cv::GaussianBlur(src, blur, kernel, sigmaX, sigmaY, borderKind);
    makeWindow("Original Blur", blur);
    cv::Mat result;
    cv::cvtColor(blur, result, cv::COLOR_RGB2GRAY);
    makeWindow("Original Blurred Grayscale", result);
    return result;
}

// Show application of Laplacian() to src.
//
static void showLaplacian(const cv::Mat &src, int kernelSize)
{
    static const int borderKind = cv::BORDER_DEFAULT;
    static const int depth = CV_16S;
    static const double scale = 1.0;
    static const double delta = 0.0;
    static const double alpha = 0.5;
    static const double beta  = 0.5;
    static const double gamma = 0.0;
    cv::Mat dst, absDst;
    cv::Laplacian(src, dst, depth, kernelSize, scale, delta, borderKind);
    cv::convertScaleAbs(dst, absDst);
    makeWindow("Laplacian", absDst);
}

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            static const int kernelSize = 3;
            const cv::Mat blurGray = showOriginalBlurGray(image, kernelSize);
            showLaplacian(blurGray, kernelSize);
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate the Laplacian operator."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
