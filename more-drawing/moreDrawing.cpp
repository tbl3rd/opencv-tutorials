/**
 * @file Drawing_2.cpp
 * @brief Simple sample code
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

static const char windowName[] = "Drawing_2 Tutorial";
static const int windowWidth = 900;
static const int windowHeight = 600;

static const int x_1 = -1 * windowWidth / 2;
static const int x_2 = +3 * windowWidth / 2;
static const int y_1 = -1 * windowWidth / 2;
static const int y_2 = +3 * windowWidth / 2;

static const int LINETYPE = 8;
static const int NUMBER = 100;
static const int DELAY = 5;

static cv::RNG rng(0xffffffff);

static cv::Scalar randomColor()
{
    return cv::Scalar((schar)rng, (schar)rng, (schar)rng);
}

static int randomLines(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point pt1(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2));
        const cv::Point pt2(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2));
        cv::line(image, pt1, pt2, randomColor(), rng.uniform(1, 10), 8);
        cv::imshow(windowName, image);
        if (cv::waitKey(DELAY) >= 0) return true;
    }
    return false;
}

static int randomRectangles(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const int thickness = MAX(rng.uniform(-3, 10), -1);
        const cv::Point pt1(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2));
        const cv::Point pt2(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2));
        cv::rectangle(image, pt1, pt2, randomColor(), thickness, LINETYPE);
        cv::imshow(windowName, image);
        if (cv::waitKey(DELAY) >= 0) return true;
    }
    return false;
}

static int randomEllipses(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point center(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2));
        const cv::Size axes(rng.uniform(0, 200), rng.uniform(0, 200));
        const double angle = rng.uniform(0, 180);
        cv::ellipse(image, center, axes, angle, angle - 100, angle + 200,
                    randomColor(), rng.uniform(-1, 9), LINETYPE);
        cv::imshow(windowName, image);
        if (cv::waitKey(DELAY) >= 0) return true;
    }
    return false;
}

static int randomPolylines(cv::Mat &image)
{
    for (int i = 0; i< NUMBER; ++i) {
        const cv::Point pt[2][3] = {
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2))
        };
        const cv::Point *ppt[2] = { pt[0], pt[1] };
        const int npt[] = {3, 3};
        cv::polylines(image, ppt, npt, 2, true,
                      randomColor(), rng.uniform(1, 10), LINETYPE);
        cv::imshow(windowName, image);
        if (cv::waitKey(DELAY) >= 0) return true;
    }
    return false;
}

static int randomFilledPolygons(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point pt[2][3] = {
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2)),
            cv::Point(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2))
        };
        const cv::Point* ppt[2] = { pt[0], pt[1] };
        const int npt[] = {3, 3};
        cv::fillPoly(image, ppt, npt, 2, randomColor(), LINETYPE);
        cv::imshow(windowName, image);
        if (cv::waitKey(DELAY) >= 0) return true;
    }
    return false;
}

static int randomCircles(cv::Mat &image)
{
    for (int i = 0; i < NUMBER; ++i) {
        const cv::Point center(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2));
        cv::circle(image, center, rng.uniform(0, 300), randomColor(),
                   rng.uniform(-1, 9), LINETYPE);
        cv::imshow(windowName, image);
        if (cv::waitKey(DELAY) >= 0) return true;
    }
    return false;
}

static int randomText(cv::Mat &image)
{
    for (int i = 1; i < NUMBER; ++i) {
        const cv::Point org(rng.uniform(x_1, x_2), rng.uniform(y_1, y_2));
        cv::putText(image, "Testing text rendering", org, rng.uniform(0, 8),
                    0.05 * rng.uniform(0, 100) + 0.1, randomColor(),
                    rng.uniform(1, 10), LINETYPE);
        cv::imshow(windowName, image);
        if (cv::waitKey(DELAY) >= 0) return true;
    }
    return false;
}

static int bigEnd(cv::Mat &image)
{
    const cv::Size textSize
        = cv::getTextSize("OpenCV forever!",
                          cv::FONT_HERSHEY_COMPLEX, 3.0, 5, 0);
    const cv::Point org((windowWidth  - textSize.width)  / 2,
                        (windowHeight - textSize.height) / 2);
    for (int i = 0; i < 255; i += 2) {
        cv::Mat image2 = image - cv::Scalar::all(i);
        cv::putText(image2, "OpenCV forever!", org,
                    cv::FONT_HERSHEY_COMPLEX, 3,
                    cv::Scalar(i, i, 255), 5, LINETYPE);
        cv::imshow(windowName, image2);
        if (cv::waitKey(DELAY) >= 0) return true;
    }
    return false;
}

int main(int, const char *[])
{
    cv::Mat image = cv::Mat::zeros(windowHeight, windowWidth, CV_8UC3);
    cv::imshow(windowName, image);
    cv::waitKey(DELAY);
    static int (*const draw[])(cv::Mat &image) = {
        &randomLines,
        &randomRectangles,
        &randomEllipses,
        &randomPolylines,
        &randomFilledPolygons,
        &randomCircles,
        &randomText,
        &bigEnd
    };
    static const int count = sizeof draw / sizeof draw[0];
    for (int i = 0; i < count; ++i) if ((*draw[i])(image)) return 0;
    cv::waitKey(0);
    return 0;
}
