#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

static const int LINETYPE = 8;
static const int NUMBER = 255;

static cv::RNG rng(0xffffffff);

// Show image and return true if program should stop.
//
static bool showImage(const cv::Mat &image)
{
    cv::imshow("Drawing_2 Tutorial", image);
    return cv::waitKey(5) >= 0;
}

// Return a random RGB color.
//
static cv::Scalar randomColor(void)
{
    const char blue  = (schar)rng;
    const char green = (schar)rng;
    const char red   = (schar)rng;
    return cv::Scalar(blue, green, red);
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
    for (int i = 0; i < NUMBER; ++i) {
        cv::line(image, randomPoint(image), randomPoint(image),
                 randomColor(), rng.uniform(1, 10), 8);
        if (showImage(image)) return true;
    }
    return false;
}

// Draw random rectangles on image.
//
static int randomRectangles(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const int thickness = MAX(rng.uniform(-3, 10), -1);
        cv::rectangle(image, randomPoint(image), randomPoint(image),
                      randomColor(), thickness, LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

// Draw random ellipses on image.
//
static int randomEllipses(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point center = randomPoint(image);
        const cv::Size axes(rng.uniform(0, 200), rng.uniform(0, 200));
        const double angle = rng.uniform(0, 180);
        cv::ellipse(image, center, axes, angle, angle - 100, angle + 200,
                    randomColor(), rng.uniform(-1, 9), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

// Draw random polylines on image.
//
static int randomPolylines(cv::Mat &image)
{
    for (int i = 0; i< NUMBER; ++i) {
        const cv::Point pt[2][3] = {
            randomPoint(image), randomPoint(image), randomPoint(image),
            randomPoint(image), randomPoint(image), randomPoint(image)
        };
        const cv::Point *ppt[2] = { pt[0], pt[1] };
        const int npt[] = {3, 3};
        cv::polylines(image, ppt, npt, 2, true,
                      randomColor(), rng.uniform(1, 10), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

// Draw random filled polygons on image.
//
static int randomFilledPolygons(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point pt[2][3] = {
            randomPoint(image), randomPoint(image), randomPoint(image),
            randomPoint(image), randomPoint(image), randomPoint(image)
        };
        const cv::Point *ppt[2] = { pt[0], pt[1] };
        const int npt[] = {3, 3};
        cv::fillPoly(image, ppt, npt, 2, randomColor(), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

// Draw random circles on image.
//
static int randomCircles(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point center = randomPoint(image);
        cv::circle(image, center, rng.uniform(0, 300), randomColor(),
                   rng.uniform(-1, 9), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

// Render text in random styles on image.
//
static int randomText(cv::Mat &image)
{
    for (int i = 1; i < NUMBER; ++i) {
        const cv::Point origin = randomPoint(image);
        cv::putText(image, "Testing text rendering", origin, rng.uniform(0, 8),
                    0.05 * rng.uniform(0, 100) + 0.1, randomColor(),
                    rng.uniform(1, 10), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

// Render a red message on image, then fade image to black while fading
// text to white.
//
static int bigFinale(cv::Mat &image)
{
    static const int font = cv::FONT_HERSHEY_COMPLEX;
    static const char msg[] = "OpenCV forever!";
    const cv::Size textSize = cv::getTextSize(msg, font, 3.0, 5, 0);
    const cv::Size size = image.size() - textSize;
    const cv::Point origin(size.width / 2, size.height / 2);
    for (int i = 0; i < NUMBER; i += 2) {
        const cv::Scalar color(i, i, 255);
        cv::Mat fade = image - cv::Scalar::all(i);
        cv::putText(fade, msg, origin, font, 3, color, 5, LINETYPE);
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
        &randomEllipses,
        &randomPolylines,
        &randomFilledPolygons,
        &randomCircles,
        &randomText,
        &bigFinale
    };
    static const int count = sizeof draw / sizeof draw[0];
    for (int i = 0; i < count; ++i) if ((*draw[i])(image)) return 0;
    cv::waitKey(0);
    return 0;
}
