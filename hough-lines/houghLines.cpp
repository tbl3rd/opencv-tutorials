#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Display image in the named window unobscured.
//
static void makeWindow(const char *window, const cv::Mat &image)
{
    static const int across = 2;
    static int moveCount = 0;
    cv::imshow(window, image);
    const int moveX = (moveCount % across) * image.cols;
    const int moveY = (moveCount / across) * (50 + image.rows);
    cv::moveWindow(window, moveX, moveY);
    ++moveCount;
}

static cv::Mat cannyDetect(const cv::Mat &image)
{
    static const double threshold1 =  50;
    static const double threshold2 = 200;
    static const int kernelSize = 3;
    static const bool l2Gradient = false;
    cv::Mat result;
    image.copyTo(result);
    cv::Canny(image, result, threshold1, threshold1, kernelSize, l2Gradient);
    return result;
}

static cv::Mat standardHough(const cv::Mat &cannyImg, const cv::Mat &colorImg)
{
    cv::Mat result;
    colorImg.copyTo(result);
    std::vector<cv::Vec2f> lines;
    static const double rho = 1;
    static const double theta = CV_PI / 180.0;
    static const int threshold = 275;
    static const double srn = 0.0;
    static const double stn = 0.0;
    cv::HoughLines(cannyImg, lines, rho, theta, threshold, srn, stn);
    const int diagonal = std::hypot(result.rows, result.cols);
    for (int i = 0; i < lines.size(); ++i) {
        const cv::Vec2f coordinate = lines[i];
        const float  rho           = coordinate[0];
        const float  theta         = coordinate[1];
        const double cosTheta      = cos(theta);
        const double sinTheta      = sin(theta);
        const cv::Point pt1(cvRound(rho * cosTheta - diagonal * sinTheta),
                            cvRound(rho * sinTheta + diagonal * cosTheta));
        const cv::Point pt2(cvRound(rho * cosTheta + diagonal * sinTheta),
                            cvRound(rho * sinTheta - diagonal * cosTheta));
        static const cv::Scalar color(0, 0, 255);
        static const int thickness = 3;
        static const int antiAliasedLine = CV_AA;
        static const int shift = 0;
        cv::line(result, pt1, pt2, color, thickness, antiAliasedLine, shift);
    }
    return result;
}

static cv::Mat probableHough(const cv::Mat &cannyImg, const cv::Mat &colorImg)
{
    cv::Mat result;
    colorImg.copyTo(result);
    std::vector<cv::Vec4i> lines;
    static const double rho = 1.0;
    static const double theta = CV_PI / 180.0;
    static const int threshold = 200;
    static const double minLength = 50.0;
    static const double maxGap = 5.0;
    cv::HoughLinesP(cannyImg, lines, rho, theta, threshold, minLength, maxGap);
    for (int i = 0; i < lines.size(); ++i) {
        cv::Vec4i coordinate = lines[i];
        const cv::Point pt1(coordinate[0], coordinate[1]);
        const cv::Point pt2(coordinate[2], coordinate[3]);
        static const cv::Scalar color(0, 0, 255);
        static const int thickness = 3;
        static const int antiAliasedLine = CV_AA;
        static const int shift = 0;
        cv::line(result, pt1, pt2, color, thickness, antiAliasedLine, shift);
    }
    return result;
}

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            makeWindow("Original", image);
            const cv::Mat cannyImage = cannyDetect(image);
            makeWindow("Canny Edges", cannyImage);
            const cv::Mat sHough = standardHough(cannyImage, image);
            makeWindow("Standard Hough", sHough);
            const cv::Mat pHough = probableHough(cannyImage, image);
            makeWindow("Probablistic Hough", pHough);
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate line finding with Hough transform."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/building.jpg"
              << std::endl << std::endl;
    return 1;
}
