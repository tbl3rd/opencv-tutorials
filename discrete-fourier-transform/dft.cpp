#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// Show image with label in an unobstructed window.
//
static void showImage(const char *label, const cv::Mat &image)
{
    static int move = 0;
    cv::imshow(label, image);
    cv::moveWindow(label, move, 0);
    move += image.cols;
}

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

// Return dftMatrix with the top-left quadrant swapped with the
// bottom-right and with the top-right quadrant swapped with the
// bottom-left.
//
static cv::Mat centerOrigin(const cv::Mat &dftMatrix)
{
    const int cols = dftMatrix.cols & -2;
    const int rows = dftMatrix.rows & -2;
    const cv::Rect crop(0, 0, cols, rows);
    const cv::Mat result = dftMatrix(crop);
    const int halfX = result.cols / 2;
    const int halfY = result.rows / 2;
    const cv::Rect tlCrop(    0,     0, halfX, halfY); // top-left
    const cv::Rect trCrop(halfX,     0, halfX, halfY); // top-right
    const cv::Rect blCrop(    0, halfY, halfX, halfY); // bottom-left
    const cv::Rect brCrop(halfX, halfY, halfX, halfY); // bottom-right
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

// Return log(1 + ||DFT(image)||)
// or     log(1 + sqrt(Real(DFT(image))^2 + Imaginary(DFT(image))^2))
// with the resulting matrix elements normalized to between 0.0 and 1.0.
//
static cv::Mat normalizedLogDiscreteFourierTransform(const cv::Mat &image)
{
    static const cv::Scalar one = cv::Scalar::all(1);
    const cv::Mat padded = padOutImage(image);
    cv::Mat complexPlane = complexify(padded);
    cv::dft(complexPlane, complexPlane);
    cv::Mat result = realify(complexPlane) + one;
    cv::log(result, result);
    cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);
    return result;
}

int main(int ac, const char *av[])
{
    const char *const filename = ac > 1 ? av[1] : "../resources/lena.jpg";
    const cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) return 1;
    showImage("Input Image", image);
    const cv::Mat nldft = normalizedLogDiscreteFourierTransform(image);
    showImage("normalized logarithmic DFT", nldft);
    const cv::Mat output = centerOrigin(nldft);
    showImage("spectrum magnitude", output);
    cv::waitKey();
    return 0;
}
