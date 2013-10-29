#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>


// Return a copy of image padded out to optimal size for a DFT.
//
static cv::Mat padOutImage(const cv::Mat &image)
{
    static const cv::Scalar zero = cv::Scalar::all(0);
    cv::Mat result;
    const int rows = cv::getOptimalDFTSize(image.rows) - image.rows;
    const int cols = cv::getOptimalDFTSize(image.cols) - image.cols;
    cv::copyMakeBorder(image, result, 0, rows, 0, cols,
                       cv::BORDER_CONSTANT, zero);
    return result;
}

// Return image embedded in the complex plane.
//
static cv::Mat complexify(const cv::Mat &image)
{
    cv::Mat result;
    cv::Mat plane[] = {
        cv::Mat_<float>(image),
        cv::Mat::zeros(image.size(), CV_32F)
    };
    const int count = sizeof plane / sizeof plane[0];
    cv::merge(plane, count, result);
    return result;
}

// Return the real part of complex.
//
static cv::Mat realify(const cv::Mat &complex)
{
    cv::Mat result;
    cv::Mat plane[] = {
        cv::Mat_<float>(complex),
        cv::Mat::zeros(complex.size(), CV_32F)
    };
    const int count = sizeof plane / sizeof plane[0];
    cv::split(complex, plane);
    cv::magnitude(plane[0], plane[1], result);
    return result;
}

// Return real with a logarithmic scale applied.
//
static cv::Mat logify(const cv::Mat &real)
{
    static const cv::Scalar one  = cv::Scalar::all(1);
    cv::Mat result = real + 1;
    cv::log(result, result);
    return result;
}

// Return logreal with the top-left quadrant swapped with the bottom-right
// and with the top-right quadrant swapped with the bottom-left.
//
static cv::Mat centerOrigin(const cv::Mat &logreal)
{
    const int cols = logreal.cols & -2;
    const int rows = logreal.rows & -2;
    const cv::Rect crop(0, 0, cols, rows);
    const cv::Mat result = logreal(crop);
    const int halfX = result.cols / 2;
    const int halfY = result.rows / 2;
    const cv::Rect tlCrop(    0,     0, halfX, halfY); // top-left
    const cv::Rect trCrop(halfX,     0, halfX, halfY); // top-right
    const cv::Rect blCrop(    0, halfY, halfX, halfY); // top-right
    const cv::Rect brCrop(halfX, halfY, halfX, halfY); // top-right
    cv::Mat tlQuadrant(result, tlCrop);
    cv::Mat trQuadrant(result, trCrop);
    cv::Mat blQuadrant(result, blCrop);
    cv::Mat brQuadrant(result, brCrop);
    cv::Mat xyQuadrant;
    tlQuadrant.copyTo(xyQuadrant);
    brQuadrant.copyTo(tlQuadrant);
    xyQuadrant.copyTo(brQuadrant);
    trQuadrant.copyTo(xyQuadrant);
    blQuadrant.copyTo(trQuadrant);
    xyQuadrant.copyTo(blQuadrant);
    return result;
}

int main(int ac, const char *av[])
{
    const char *const filename = ac > 1 ? av[1] : "../resources/lena.jpg";
    const cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) return -1;
    cv::imshow("Input Image", image);
    const cv::Mat padded = padOutImage(image);
    cv::imshow("Padded Image", padded);
    cv::Mat complexPlane = complexify(padded);
    cv::dft(complexPlane, complexPlane);
    const cv::Mat real = realify(complexPlane);
    const cv::Mat logreal = logify(real);
    const cv::Mat normalizedLogreal = logreal;
    cv::normalize(logreal, normalizedLogreal, 0, 1, CV_MINMAX);
    cv::imshow("normalized logreal", normalizedLogreal);
    const cv::Mat output = centerOrigin(logreal);
    const cv::Mat normalizedOutput = output;
    cv::normalize(output, normalizedOutput, 0, 1, CV_MINMAX);
    cv::imshow("Input Image"       , image);
    cv::imshow("spectrum magnitude", normalizedOutput);
    cv::waitKey();
    return 0;
}
