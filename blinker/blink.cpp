#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

static void usage(const char *av0)
{
    std::cerr << av0 << ": Alternate 2 images to spot differences."
              << std::endl << std::endl
              << "Usage: " << av0 << " <ms> <left> <right" << std::endl
              << std::endl
              << "Where: <ms> is the display time im milliseconds." << std::endl
              << "       <left> is an image file." << std::endl
              << "       <right> is another image file." << std::endl
              << std::endl
              << "Example: " << av0 << " 500 left.png right.png"
              << std::endl << std::endl;
}

int main (int ac, char *av[])
{
    if (ac == 4) {
        int msDelay = 0;
        std::istringstream iss(av[1]); iss >> msDelay;
        const cv::Mat left  = cv::imread(av[2]);
        const cv::Mat right = cv::imread(av[3]);
        if (msDelay && left.data && right.data) {
            std::cout << av[0] << ": Press some key to quit." << std::endl;
            cv::namedWindow("Blink", cv::WINDOW_AUTOSIZE);
            while (true) {
                cv::imshow("Blink", left);
                cv::waitKey(msDelay);
                cv::imshow("Blink", right);
                const int c = cv::waitKey(msDelay);
                if (c != -1) break;
            }
            return 0;
        }
    }
    usage(av[0]);
    return 1;
}
