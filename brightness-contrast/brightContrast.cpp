#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

// Show a usage message on cout for program named av0.
//
static void showUsage(const char *av0)
{
    const char one[] = "../resources/LinuxLogo.jpg";
    const char two[] = "../resources/WindowsLogo.jpg";
    std::cout << std::endl
              << av0 << ": Blend two images."
              << std::endl << std::endl
              << "Usage: " << av0 << " <image-file>"
              << std::endl << std::endl
              << "Where: <image-file> is the path to an image file."
              << std::endl << std::endl
              << "Example: " << av0 << " ../resources/lena.jpg"
              << std::endl << std::endl;
}

// Return the image specified on the command line.
//
static cv::Mat useCommandLine(int ac, const char *av[])
{
    cv::Mat result;
    if (ac == 2) result = cv::imread(av[1]);
    if (!result.data) showUsage(av[0]);
    return result;
}

static cv::Mat gainBias(const cv::Mat &input, double alpha, int beta)
{
    cv::Mat_<cv::Vec3b> result = cv::Mat::zeros(input.size(), input.type());
    const cv::Mat_<cv::Vec3b> head = input;
    for (int i = 0; i < head.rows; ++i) {
        for (int j = 0; j < head.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                const int resultAtIjc = alpha * head(i,j)[c] + beta;
                result(i,j)[c] = cv::saturate_cast<uchar>(resultAtIjc);
            }
        }
    }
    return result;
}

int main(int ac, const char *av[])
{
    const cv::Mat input = useCommandLine(ac, av);
    if (!input.data) return 1;
    cv::imshow(av[1], input); cv::waitKey(50);
    static const int max = 10;
    static const double alphaMax = 2.0;
    static const int betaMax = 100;
    for (int i = 0; i <= max; ++i) {
        const double alpha = 1.0 + alphaMax * i / max;
        for (int j = 0; j <= max; ++j) {
            const int beta = 0 + betaMax * j / max;
            std::stringstream ss; ss << alpha << ":" << beta << std::ends;
            cv::Mat gb = gainBias(input, alpha, beta);
            cv::imshow(ss.str(), gb); cv::waitKey(50);
        }
    }
    cv::waitKey(0);
    return 0;
}
