#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Create a new unobscured named window for image.
// Reset windows layout with when reset is not 0.
//
// The fudge works around how MacOSX lays out window decorations.
//
static void makeWindow(const char *window, const cv::Mat &image, int reset = 0)
{
    static int across = 1;
    static int moveCount = 0;
    if (reset) {
        across = reset;
        moveCount = 0;
    }
    const int overCount = moveCount % across;
    const int downCount = moveCount / across;
    const int moveX = overCount * image.cols;
    const int moveY = downCount * image.rows;
    const int fudge = downCount == 0 ? 0 : (1 + downCount);
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window, moveX, moveY + 23 * fudge);
    cv::imshow(window, image);
    ++moveCount;
}

// Return a copy of src warped by mapping srcTri to dstTri.
//
static cv::Mat warpImage(const cv::Mat &src)
{
    const cv::Point2f srcTri[] = {
        cv::Point2f(0,            0           ),
        cv::Point2f(src.cols - 1, 0           ),
        cv::Point2f(0,            src.rows - 1)
    };
    const cv::Point2f dstTri[] = {
        cv::Point2f(src.cols * 0.00, src.rows * 0.33),
        cv::Point2f(src.cols * 0.85, src.rows * 0.25),
        cv::Point2f(src.cols * 0.15, src.rows * 0.70)
    };
    const cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
    cv::Mat result;
    cv::warpAffine(src, result, warpMat, result.size());
    return result;
}

// Return a copy of src scaled and rotated by angle about center.
//
static cv::Mat rotateImage(const cv::Mat &src)
{
    static const double angle = -50.0;
    static const double scale = 0.6;
    const cv::Point center(src.cols / 2, src.rows / 2);
    const cv::Mat rotateMat = cv::getRotationMatrix2D(center, angle, scale);
    cv::Mat result;
    cv::warpAffine(src, result, rotateMat, src.size());
    return result;
}

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat src = cv::imread(av[1]);
        if (src.data) {
            std::cout << av[0] << ": Press some key to quit." << std::endl;
            const cv::Mat warpDst = warpImage(src);
            const cv::Mat warpRotateDst = rotateImage(warpDst);
            makeWindow("Source image", src);
            makeWindow("Warp", warpDst);
            makeWindow("Warp+Rotate", warpRotateDst);
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate affine transformations."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
