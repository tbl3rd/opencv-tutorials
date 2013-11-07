#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


cv::Mat mapX, mapY;

void updateMaps(const cv::Mat &image)
{
    static int index = 0;
    index %= 5;
    const int minCols = image.cols / 4;
    const int maxCols = 3 * minCols;
    const int minRows = image.rows / 4;
    const int maxRows = 3 * minRows;
    for (int j = 0; j < image.rows; ++j) {
        for (int i = 0; i < image.cols; ++i) {
            float x = 0.0;
            float y = 0.0;
            switch (index) {
            case 0:                     // identity
                x = i;
                y = j;
                break;
            case 1: {                   // scale down / 4
                const bool ok
                    =  i < maxCols && i > minCols
                    && j < maxRows && j > minRows;
                if (ok) {
                    x = 0.5 + 2 * (i - minCols);
                    y = 0.5 + 2 * (j - minRows);
                }
                break;
            }
            case 2:                     // mirror horizontal
                x = i ;
                y = image.rows - j ;
                break;
            case 3:                     // mirror vertical
                x = image.cols - i ;
                y = j ;
                break;
            case 4:                     // rotate 180
                x = image.cols - i ;
                y = image.rows - j ;
                break;
            }
            mapX.at<float>(j, i) = x;
            mapY.at<float>(j, i) = y;
        }
    }
    ++index;
}

int main(int ac, const char *av[])
{
    static const int interpolation = cv::INTER_LINEAR;
    static const int borderKind = cv::BORDER_CONSTANT;
    static const cv::Scalar borderValue(0, 0, 0);
    static const int waitMilliseconds = 1000;
    if (ac == 2) {
        const cv::Mat src = cv::imread(av[1]);
        if (src.data) {
            cv::Mat dst(src.size(), src.type());
            mapX.create(src.size(), CV_32FC1);
            mapY.create(src.size(), CV_32FC1);
            while (true) {
                updateMaps(src);
                cv::remap(src, dst, mapX, mapY, interpolation,
                          borderKind, borderValue);
                cv::imshow("Remap demo", dst);
                const int c = cv::waitKey(waitMilliseconds);
                if('Q' == c || 'q' == c) break;
            }
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate image remapping."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/prototype.jpg"
              << std::endl << std::endl;
    return 1;
}
