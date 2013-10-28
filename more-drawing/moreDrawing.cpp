/**
 * @file Drawing_2.cpp
 * @brief Simple sample code
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

static const int windowWidth = 900;
static const int windowHeight = 600;

static const int LINETYPE = 8;
static const int NUMBER = 255;

static cv::RNG rng(0xffffffff);

static bool showImage(const cv::Mat &image)
{
    cv::imshow("Drawing_2 Tutorial", image);
    return cv::waitKey(5) >= 0;
}

static cv::Scalar randomColor(void)
{
    const char blue  = (schar)rng;
    const char green = (schar)rng;
    const char red   = (schar)rng;
    return cv::Scalar(blue, green, red);
}

static cv::Point randomPoint(void)
{
    static const int x_1 = -1 * windowWidth  / 2;
    static const int x_2 = +3 * windowWidth  / 2;
    static const int y_1 = -1 * windowHeight / 2;
    static const int y_2 = +3 * windowHeight / 2;
    const int x = rng.uniform(x_1, x_2);
    const int y = rng.uniform(y_1, y_2);
    const cv::Point result(x, y);
    return result;
}

static int randomLines(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        cv::line(image, randomPoint(), randomPoint(), randomColor(),
                 rng.uniform(1, 10), 8);
        if (showImage(image)) return true;
    }
    return false;
}

static int randomRectangles(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const int thickness = MAX(rng.uniform(-3, 10), -1);
        cv::rectangle(image, randomPoint(), randomPoint(), randomColor(),
                      thickness, LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

static int randomEllipses(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point center = randomPoint();
        const cv::Size axes(rng.uniform(0, 200), rng.uniform(0, 200));
        const double angle = rng.uniform(0, 180);
        cv::ellipse(image, center, axes, angle, angle - 100, angle + 200,
                    randomColor(), rng.uniform(-1, 9), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

static int randomPolylines(cv::Mat &image)
{
    for (int i = 0; i< NUMBER; ++i) {
        const cv::Point pt[2][3] = {
            randomPoint(), randomPoint(), randomPoint(),
            randomPoint(), randomPoint(), randomPoint()
        };
        const cv::Point *ppt[2] = { pt[0], pt[1] };
        const int npt[] = {3, 3};
        cv::polylines(image, ppt, npt, 2, true,
                      randomColor(), rng.uniform(1, 10), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

static int randomFilledPolygons(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point pt[2][3] = {
            randomPoint(), randomPoint(), randomPoint(),
            randomPoint(), randomPoint(), randomPoint()
        };
        const cv::Point *ppt[2] = { pt[0], pt[1] };
        const int npt[] = {3, 3};
        cv::fillPoly(image, ppt, npt, 2, randomColor(), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

static int randomCircles(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point center = randomPoint();
        cv::circle(image, center, rng.uniform(0, 300), randomColor(),
                   rng.uniform(-1, 9), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

static int randomText(cv::Mat &image)
{
    for (int i = 1; i < NUMBER; ++i) {
        const cv::Point origin = randomPoint();
        cv::putText(image, "Testing text rendering", origin, rng.uniform(0, 8),
                    0.05 * rng.uniform(0, 100) + 0.1, randomColor(),
                    rng.uniform(1, 10), LINETYPE);
        if (showImage(image)) return true;
    }
    return false;
}

static int bigFinale(cv::Mat &image)
{
    static const int font = cv::FONT_HERSHEY_COMPLEX;
    static const char msg[] = "OpenCV forever!";
    const cv::Size textSize = cv::getTextSize(msg, font, 3.0, 5, 0);
    const cv::Point origin((windowWidth  - textSize.width)  / 2,
                        (windowHeight - textSize.height) / 2);
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
    cv::Mat image = cv::Mat::zeros(windowHeight, windowWidth, CV_8UC3);
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
