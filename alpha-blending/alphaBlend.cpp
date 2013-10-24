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
        << "Usage: " << av0 << " <image-file-1> <image-file-2>"
        << std::endl << std::endl
        << "Where: <image-file-1> is the path to an image file."
        << std::endl
        << "       <image-file-2> is the path to an image file."
        << std::endl
        << "       And both image files have the same size and type."
        << std::endl << std::endl
        << "Example: " << av0 << " " << one << " " << two
        << std::endl << std::endl;
}

// Return 1 with the images specified on the command line in one and two.
// Otherwise show the usage message and return 0.
//
static int useCommandLine(int ac, const char *av[],
                          cv::Mat &one, cv::Mat &two)
{
    int ok = ac == 3;
    if (ok) {
        one = cv::imread(av[1]);
        ok = !!one.data;
        if (ok) {
            two = cv::imread(av[2]);
            ok = !!two.data;
            if (ok) {
                ok =   one.type() == two.type()
                    && one.rows   == two.rows
                    && one.cols   == two.cols;
            }
        }
    }
    if (!ok) showUsage(av[0]);
    return ok;
}

int main(int ac, const char *av[])
{
    cv::Mat one, two;
    const int ok = useCommandLine(ac, av, one, two);
    if (!ok) return 1;
    cv::imshow(av[1], one); cv::waitKey(50);
    cv::imshow(av[2], two); cv::waitKey(50);
    static const int max = 10;
    for (int i = 0; i <= max; ++i) {
        const double alpha = 0.0 + ((double)i / max);
        cv::Mat blend;
        cv::addWeighted(one, alpha, two, 1.0 - alpha, 0.0, blend);
        std::stringstream ss; ss << alpha << std::ends;
        cv::imshow(ss.str(), blend); cv::waitKey(50);
    }
    cv::waitKey(0);
    return 0;
}
