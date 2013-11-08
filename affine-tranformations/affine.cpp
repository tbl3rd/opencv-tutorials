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


static void showTransformations(const cv::Mat &src)
{
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];

    cv::Mat rot_mat(2, 3, CV_32FC1);
    cv::Mat warp_mat(2, 3, CV_32FC1);
    cv::Mat warp_dst, warp_rotate_dst;

    /// Set the dst image the same type and size as src
    warp_dst = cv::Mat::zeros(src.rows, src.cols, src.type());

    /// Set your 3 points to calculate the  Affine Transform
    srcTri[0] = cv::Point2f(0,0);
    srcTri[1] = cv::Point2f(src.cols - 1, 0);
    srcTri[2] = cv::Point2f(0, src.rows - 1);

    dstTri[0] = cv::Point2f(src.cols * 0.0, src.rows * 0.33);
    dstTri[1] = cv::Point2f(src.cols * 0.85, src.rows * 0.25);
    dstTri[2] = cv::Point2f(src.cols * 0.15, src.rows * 0.7);

    /// Get the Affine Transform
    warp_mat = cv::getAffineTransform(srcTri, dstTri);

    /// Apply the Affine Transform just found to the src image
    cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());

    /** Rotating the image after Warp */

    /// Compute a rotation matrix with respect to the center of the image
    cv::Point center(warp_dst.cols / 2, warp_dst.rows / 2);
    double angle = -50.0;
    double scale = 0.6;

    /// Get the rotation matrix with the specifications above
    rot_mat = cv::getRotationMatrix2D(center, angle, scale);

    /// Rotate the warped image
    cv::warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());

    /// Show what you got
    makeWindow("Source image", src);
    makeWindow("Warp", warp_dst);
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
