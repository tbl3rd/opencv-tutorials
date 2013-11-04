#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Write image and return true if the command line at (ac, av) is OK.
// Otherwise show a usage message and return false.
//
static bool commandLineOk(int ac, const char *av[], cv::Mat &image)
{
    if (ac == 2) {
        const cv::Mat av1Image = cv::imread(av[1]);
        const bool ok = av1Image.data
            && (0 == (av1Image.cols % 2))
            && (0 == (av1Image.rows % 2));
        if (ok) {
            image = av1Image;
            return true;
        }
    }
    std::cerr << av[0] << ": Demonstrate image pyramids." << std::endl
              << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/chicky_512.jpg"
              << std::endl << std::endl;
    return false;
}


int main(int ac, const char *av[])
{
    cv::Mat src;
    if (commandLineOk(ac, av, src)) {
        std::cout << av[0] << ": Use the following keys in the demo window."
                  << std::endl
                  << "  u  -=>  Zoom up or in."         << std::endl
                  << "  d  -=>  Zoom down or out."      << std::endl
                  << "  q  -=>  Close window and quit." << std::endl
                  << std::endl;
        cv::namedWindow("Pyramids Demo", cv::WINDOW_AUTOSIZE);
        while (true) {
            cv::Mat dst = src;
            cv::imshow("Pyramids Demo", dst);
            switch (cv::waitKey(10)) {
            case 'Q': case 'q':
                std::cout << av[0] << ": Quitting now." << std::endl;
                return 0;
                break;
            case 'U': case 'u':
                std::cout << av[0] << ": Zooming in  * 2" << std::endl;
                cv::pyrUp(src, dst, cv::Size(2 * src.cols, 2 * src.rows));
                break;
            case 'D': case 'd':
                std::cout << av[0] << ": Zooming out / 2" << std::endl;
                cv::pyrDown(src, dst, cv::Size(src.cols / 2, src.rows / 2));
                break;
            }
            src = dst;
        }
        return 0;
    }
    return 1;
}
