#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

// Show a usage message on cout for program named av0.
//
static void showUsage(const char *av0)
{
    std::cout
        << av0 << ": Filter an image with a 'sharpening' mask."
        << std::endl << std::endl
        << "Usage: " << av0 << " <image-file> [g]"
        << std::endl << std::endl
        << "Where: <image-file> is the path to an image file."
        << std::endl
        << "       The image should have a Mat::depth() of CV_8U."
        << std::endl
        << "       g means process the image in gray scale."
        << std::endl << std::endl
        << "Example: " << av0 << " ../resources/lena.tiff"
        << std::endl
        << "Read an image object from lena.tiff into a cv::Mat."
        << std::endl
        << "Repeatedly sharpen the image by applying a mask cv::Mat."
        << std::endl << std::endl;
}

// Return the image specified on the command line.
// Show the usage message if something is wrong.
//
static cv::Mat useCommandLine(int ac, const char *av[])
{
    cv::Mat result;
    int ok = ac == 2 || ac == 3;
    if (ok) {
        const bool g = (ac == 3) && (*"g" == *av[2]);
        const int cogOpt = g? CV_LOAD_IMAGE_GRAYSCALE: CV_LOAD_IMAGE_COLOR;
        result = cv::imread(av[1], cogOpt);
        ok = CV_8U == result.depth();
    }
    if (!ok) showUsage(av[0]);
    return result;
}

static cv::Mat sharpen(const cv::Mat &image)
{
    cv::Mat result(image.size(), image.type());
    const int nChannels = image.channels();
    const int rowMax = image.rows - 1;
    const int colMax = image.cols - 1;
    for (int j = 1 ; j < rowMax; ++j) {
        const uchar *const previous = image.ptr<uchar>(j - 1);
        const uchar *const current  = image.ptr<uchar>(j    );
        const uchar *const next     = image.ptr<uchar>(j + 1);
        uchar *output = result.ptr<uchar>(j);
        for (int i= nChannels; i < nChannels * colMax; ++i) {
            const uchar sharper = 0
                - previous[i]
                - current[i - nChannels]
                + 5 * current[i]
                - current[i + nChannels]
                - next[i];
            *output++ = cv::saturate_cast<uchar>(sharper);
        }
    }
    static const cv::Scalar zero(0);
    result.row(0).setTo(zero);
    result.row(result.rows - 1).setTo(zero);
    result.col(0).setTo(zero);
    result.col(result.cols - 1).setTo(zero);
    return result;
}

int main(int ac, const char *av[])
{
    const cv::Mat inputImage = useCommandLine(ac, av);
    if (!inputImage.data) return 1;
    cv::namedWindow("Input", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Output", CV_WINDOW_AUTOSIZE);
    cv::imshow("Input", inputImage);

    int64 tickZero = cv::getTickCount();
    const cv::Mat useSharpenFunction = sharpen(inputImage);
    int64 ticks = cv::getTickCount() - tickZero;
    double time = (double)ticks / cv::getTickFrequency();
    std::cout << "Hand written function times passed in seconds: "
              << time << std::endl;
    cv::imshow("Output", useSharpenFunction);
    cv::waitKey(0);
    const cv::Mat kern = (cv::Mat_<char>(3,3) <<
                          +0, -1, +0,
                          -1, +5, -1,
                          +0, -1, +0);
    cv::Mat useFilter2d;
    tickZero = cv::getTickCount();
    cv::filter2D(inputImage, useFilter2d, inputImage.depth(), kern);
    ticks = cv::getTickCount() - tickZero;
    time = (double)ticks / cv::getTickFrequency();
    std::cout << "Built-in filter2D time passed in seconds:      "
              << time << std::endl;
    cv::imshow("Output", useFilter2d);
    cv::waitKey(0);
    return 0;
}
