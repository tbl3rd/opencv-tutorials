#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

// A single source of randomness for this program.
//
static cv::RNG rng(0xffffffff);

// Show image and return true if program should stop.
//
static bool showImage(const cv::Mat &image)
{
    cv::imshow("Drawing_2 Tutorial", image);
    return cv::waitKey(5) >= 0;
}

// The default line type is 8 which means 8-connected Bresenham.
// 4 means use a 4-connected Bresenham algorithm.
// CV_AA means anti-alias the line.
//
static int randomLineType(void)
{
    static const int linetype[] = { 8, 4, CV_AA };
    static const int count = sizeof linetype / sizeof linetype[0];
    const int index = rng.uniform(0, count);
    return linetype[index];
}

// Return a random RGB color.
//
static cv::Scalar randomColor(void)
{
    const char blue  = schar(rng);
    const char green = schar(rng);
    const char red   = schar(rng);
    return cv::Scalar(blue, green, red);
}

// Return a random font face.  This is only for documentation because for
// all i, i == face[i].  This could just return the index, in other words.
//
static int randomFontFace(void)
{
    static const int face[] = {
        [0] = cv::FONT_HERSHEY_SIMPLEX,
        [1] = cv::FONT_HERSHEY_PLAIN,
        [2] = cv::FONT_HERSHEY_DUPLEX,
        [3] = cv::FONT_HERSHEY_COMPLEX,
        [4] = cv::FONT_HERSHEY_TRIPLEX,
        [5] = cv::FONT_HERSHEY_COMPLEX_SMALL,
        [6] = cv::FONT_HERSHEY_SCRIPT_SIMPLEX,
        [7] = cv::FONT_HERSHEY_SCRIPT_COMPLEX
    };
    static const int count = sizeof face / sizeof face[0];
    const int index = rng.uniform(0, count);
    return face[index];
}

// Return a random point from a larger region centered around image.
//
static cv::Point randomPoint(const cv::Mat &image)
{
    const cv::Size size = image.size();
    const int x_1 = -1 * size.width  / 2;
    const int x_2 = +3 * size.width  / 2;
    const int y_1 = -1 * size.height / 2;
    const int y_2 = +3 * size.height / 2;
    const int x = rng.uniform(x_1, x_2);
    const int y = rng.uniform(y_1, y_2);
    const cv::Point result(x, y);
    return result;
}

// Draw random lines on image.
//
static int randomLines(cv::Mat &image)
{
    for (int i = 0; i < UCHAR_MAX; ++i) {
        const int thickness = rng.uniform(1, 10);
        cv::line(image, randomPoint(image), randomPoint(image),
                 randomColor(), thickness, randomLineType());
        if (showImage(image)) return true;
    }
    return false;
}

// Draw random rectangles on image.  Fill about 1/3 of them.
//
static int randomRectangles(cv::Mat &image)
{
    for (int i = 0; i < UCHAR_MAX; ++i) {
        const int thickness = MAX(rng.uniform(-3, 10), CV_FILLED);
        cv::rectangle(image, randomPoint(image), randomPoint(image),
                      randomColor(), thickness, randomLineType());
        if (showImage(image)) return true;
    }
    return false;
}

// Draw random 5/6 open elliptical arcs (some filled) on image.
//
static int randomEllipticArcs(cv::Mat &image)
{
    for (int i = 0; i < UCHAR_MAX; ++i) {
        const cv::Point center = randomPoint(image);
        const cv::Size axes(rng.uniform(0, 200), rng.uniform(0, 200));
        const double angle = rng.uniform(0, 180);
        const int thickness = rng.uniform(-1, 9);
        cv::ellipse(image, center, axes, angle, angle - 100, angle + 200,
                    randomColor(), thickness, randomLineType());
        if (showImage(image)) return true;
    }
    return false;
}

// Draw two random triangles of varying thickness on image.
//
static int randomTriangles(cv::Mat &image)
{
    static const int vertexCount = 3;
    static const int polyCount = 2;
    for (int i = 0; i< UCHAR_MAX; ++i) {
        const cv::Point points[polyCount][vertexCount] = {
            randomPoint(image), randomPoint(image), randomPoint(image),
            randomPoint(image), randomPoint(image), randomPoint(image)
        };
        const cv::Point *curves[polyCount] = { points[0], points[1] };
        const int vertexCounts[polyCount] = { vertexCount, vertexCount };
        const int thickness = rng.uniform(1, 10);
        cv::polylines(image, curves, vertexCounts, polyCount, true,
                      randomColor(), rng.uniform(1, 10), randomLineType());
        if (showImage(image)) return true;
    }
    return false;
}

// Draw two random filled triangles on image.
//
static int randomFilledTriangles(cv::Mat &image)
{
    static const int vertexCount = 3;
    static const int polyCount = 2;
    for (int i = 0; i < UCHAR_MAX; ++i) {
        const cv::Point points[polyCount][vertexCount] = {
            randomPoint(image), randomPoint(image), randomPoint(image),
            randomPoint(image), randomPoint(image), randomPoint(image)
        };
        const cv::Point *polys[polyCount] = { points[0], points[1] };
        const int vertexCounts[polyCount] = { vertexCount, vertexCount };
        cv::fillPoly(image, polys, vertexCounts, 2, randomColor(),
                     randomLineType());
        if (showImage(image)) return true;
    }
    return false;
}

// Draw random circles on image.
//
static int randomCircles(cv::Mat &image)
{
    for (int i = 0; i < UCHAR_MAX; ++i) {
        const cv::Point center = randomPoint(image);
        const int radius = rng.uniform(0, 300);
        const int thickness = rng.uniform(-1, 9);
        cv::circle(image, center, radius, randomColor(), thickness,
                   randomLineType());
        if (showImage(image)) return true;
    }
    return false;
}

// Render text in random styles at random points on image.
//
static int randomText(cv::Mat &image)
{
    static const char msg[] = "Testing text rendering";
    for (int i = 1; i < UCHAR_MAX; ++i) {
        const cv::Point origin = randomPoint(image);
        const double scale = 0.1 + 0.05 * rng.uniform(0, 100);
        const int thickness = rng.uniform(1, 10);
        cv::putText(image, msg, origin, randomFontFace(), scale,
                    randomColor(), thickness, randomLineType());
        if (showImage(image)) return true;
    }
    return false;
}

// Render a red message on image, then fade image to black while fading
// text to white.
//
static int bigFinale(cv::Mat &image)
{
    static const char msg[] = "OpenCV forever!";
    static const int face = cv::FONT_HERSHEY_COMPLEX;
    static const double scale = 3.0;
    static const int thickness = 5;
    static const cv::Size msgSize
        = cv::getTextSize(msg, face, scale, thickness, 0);
    const cv::Size size = image.size() - msgSize;
    const cv::Point origin(size.width / 2, size.height / 2);
    for (int i = 0; i < UCHAR_MAX; ++i) {
        const cv::Scalar color(i, i, 255);
        cv::Mat fade = image - cv::Scalar::all(i);
        cv::putText(fade, msg, origin, face, scale, color, thickness,
                    randomLineType());
        if (showImage(fade)) return true;
    }
    return false;
}

int main(int, const char *[])
{
    cv::Mat image = cv::Mat::zeros(600, 900, CV_8UC3);
    if (showImage(image)) return 0;
    static int (*const draw[])(cv::Mat &image) = {
        &randomLines,
        &randomRectangles,
        &randomEllipticArcs,
        &randomTriangles,
        &randomFilledTriangles,
        &randomCircles,
        &randomText,
        &bigFinale
    };
    static const int count = sizeof draw / sizeof draw[0];
    for (int i = 0; i < count; ++i) if ((*draw[i])(image)) return 0;
    cv::waitKey(0);
    return 0;
}
