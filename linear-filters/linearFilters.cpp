#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Return a normalized box filter kernel of size i for filter2D().
//
static cv::Mat makeKernel(int i)
{
    const int size = 3 + 2 * (i % 99);
    const float scale = size * size;
    return cv::Mat::ones(size, size, CV_32F) / scale;
}

// Return kernel applied to src using filter2D() showing defaults.
//
static cv::Mat applyFilter(const cv::Mat &src, const cv::Mat &kernel)
{
    static const int depth = -1;
    static const cv::Point anchor(-1, -1);
    static const double delta = 0.0;
    static const int border = cv::BORDER_DEFAULT;
    cv::Mat result;
    cv::filter2D(src, result, depth , kernel, anchor, delta, border);
    return result;
}

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat src = cv::imread(av[1]);
        if (src.data) {
            cv::namedWindow("filter2d() demo", cv::WINDOW_AUTOSIZE);
            std::cout << av[0] << ": Press some key to quit." << std::endl;
            for (int i = 0; i < std::numeric_limits<int>::max(); ++i) {
                const cv::Mat kernel = makeKernel(i);
                const cv::Mat dst = applyFilter(src, kernel);
                cv::imshow("filter2d() demo", dst);
                const int c = cv::waitKey(500);
                if (c != -1) break;
            }
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate a custom 2d linear convolution."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/mandrill.tiff"
              << std::endl << std::endl;
    return 1;
}
