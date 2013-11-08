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
    const int overCount = moveCount % across;
    const int downCount = moveCount / across;
    const int moveX = overCount * image.cols;
    const int moveY = downCount * image.rows;
    const int fudge = downCount == 0 ? 0 : (1 + downCount);
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window, moveX, moveY + 23 * fudge);
    cv::imshow(window, image);
    ++moveCount;
}

static void showHistogram(const cv::Mat &image)
{
    static const cv::Mat noArray;
    static const int     imageCount     = 1;
    static const int     dimensionCount = 1;
    static const int     histSize       = 256;
    static const int     histSizes[]    = { histSize };
    static const float   histRange[]    = { 0, histSize };
    static const float  *histRanges[]   = { histRange };
    static const bool    uniform        = true;
    static const bool    accumulate     = false;
    enum Color { BLUE, GREEN, RED, COLORCOUNT };
    cv::Mat             plane[COLORCOUNT];
    cv::Mat_<float> histogram[COLORCOUNT];
    const cv::Scalar    color[COLORCOUNT] = {
        cv::Scalar(255,   0,   0),      // BLUE
        cv::Scalar(  0, 255,   0),      // GREEN
        cv::Scalar(  0,   0, 255)       // RED
    };
    cv::split(image, plane);
    for (int c = 0; c < COLORCOUNT; ++c) {
        cv::calcHist(&plane[c], imageCount, 0, noArray, histogram[c],
                     dimensionCount, histSizes, histRanges,
                     uniform, accumulate);
    }
    static const int histWidth = 512;
    static const int histHeight = 400;
    static const double fHistWidth = histWidth;
    const int binWidth = cvRound(fHistWidth / histSizes[0]);
    cv::Mat histImage = cv::Mat_<cv::Vec3b>::zeros(histHeight, histWidth);
    for (int c = 0; c < COLORCOUNT; ++c) {
        static const double alpha = 0;
        static const int normKind = cv::NORM_MINMAX;
        static const cv::Mat mask(noArray);
        static const int dtype = -1;
        const cv::Mat_<float> &h = histogram[c];
        const double beta = histImage.rows;
        cv::normalize(h, h, alpha, beta, normKind, dtype, mask);
    }
    for (int c = 0; c < COLORCOUNT; ++c) {
        const cv::Mat_<float> &h = histogram[c];
        cv::Point p0(0, histHeight - cvRound(h(0)));
        for (int i = 1; i < histSizes[0]; ++i) {
            static const int thickness = 2;
            static const int lineType = 8;
            static const int shift = 0;
            const cv::Point p1(i * binWidth, histHeight - cvRound(h(i)));
            cv::line(histImage, p0, p1, color[c], thickness, lineType, shift);
            p0 = p1;
        }
    }
    makeWindow("Color Histogram", histImage);
}

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            std::cout << av[0] << ": Press some key to quit." << std::endl;
            cv::Mat gray, equalized;
            makeWindow("Source Image", image);
            showHistogram(image);
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate histogram equalization."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
