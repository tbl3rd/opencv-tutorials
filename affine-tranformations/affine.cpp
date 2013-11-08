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
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    const int overCount = moveCount % across;
    const int downCount = moveCount / across;
    const int moveX = overCount * image.cols;
    const int moveY = downCount * image.rows;
    const int fudge = downCount == 0 ? 0 : (1 + downCount);
    cv::moveWindow(window, moveX, moveY + 23 * fudge);
    cv::imshow(window, image);
    ++moveCount;
}


static cv::Mat warpImage(const cv::Mat &src)
{
    cv::Mat result = cv::Mat::zeros(src.size(), src.type());
    const cv::Point2f srcTri[3] = {
        cv::Point2f(0,            0           ),
        cv::Point2f(src.cols - 1, 0           ),
        cv::Point2f(0,            src.rows - 1)
    };
    const cv::Point2f dstTri[3] = {
        cv::Point2f(src.cols * 0.00, src.rows * 0.33),
        cv::Point2f(src.cols * 0.85, src.rows * 0.25),
        cv::Point2f(src.cols * 0.15, src.rows * 0.70)
    };
    const cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
    cv::warpAffine(src, result, warpMat, result.size());
    return result;
}


static void showTransformations(const cv::Mat &src)
{
    cv::Mat rot_mat(2, 3, CV_32FC1);
    cv::Mat warp_rotate_dst;

    /// Set the dst image the same type and size as src
    const cv::Mat warpDst = warpImage(src);

    /** Rotating the image after Warp */

    /// Compute a rotation matrix with respect to the center of the image
    cv::Point center(warpDst.cols / 2, warpDst.rows / 2);
    double angle = -50.0;
    double scale = 0.6;

    /// Get the rotation matrix with the specifications above
    rot_mat = cv::getRotationMatrix2D(center, angle, scale);

    /// Rotate the warped image
    cv::warpAffine(warpDst, warp_rotate_dst, rot_mat, warpDst.size());

    /// Show what you got
    makeWindow("Source image", src);
    makeWindow("Warp", warpDst);
    makeWindow("Warp+Rotate", warp_rotate_dst);
}


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat src = cv::imread(av[1]);
        if (src.data) {
            std::cout << av[0] << ": Press some key to quit." << std::endl;
            showTransformations(src);
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
