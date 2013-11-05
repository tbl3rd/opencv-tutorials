#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Return a random uchar scalar.
//
static cv::Scalar randomScalar(void)
{
    static cv::RNG rng;
    static const uchar max = std::numeric_limits<uchar>::max();
    const cv::Scalar result(rng.uniform(0, max),
                            rng.uniform(0, max),
                            rng.uniform(0, max));
    return result;
}

// Run the border demo on src for program named av0.
//
static void demoBorders(const char *av0, const cv::Mat &src)
{
    const int top  = 5 * src.rows / 100;
    const int left = 5 * src.cols / 100;
    const int bottom = top;
    const int right  = left;
    int kind = cv::BORDER_CONSTANT;
    std::cout << av0 << ": Random border." << std::endl;
    while (true) {
        const cv::Scalar value = randomScalar();
        switch (cv::waitKey(500)) {
        case 'Q': case 'q':
            std::cout << av0 << ": Quitting now." << std::endl;
            return;
        case 'C': case 'c':
            kind = cv::BORDER_CONSTANT;
            std::cout << av0 << ": Random border." << std::endl;
            break;
        case 'R': case 'r':
            kind = cv::BORDER_REPLICATE;
            std::cout << av0 << ": Replicated border." << std::endl;
            break;
        }
        cv::Mat dst;
        cv::copyMakeBorder(src, dst, top, bottom, left, right, kind, value);
        cv::imshow("copyMakeBorder() Demo", dst);
    }
}

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat src = cv::imread(av[1]);
        if (src.data) {
            std::cout << av[0] << ": copyMakeBorder() Demo"        << std::endl
                      << "Press 'c' for a random constant border." << std::endl
                      << "Press 'r' for a replicated border."      << std::endl
                      << "Press 'q' to quit the program."          << std::endl
                      << std::endl;
            demoBorders(av[0], src);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate image borders."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
