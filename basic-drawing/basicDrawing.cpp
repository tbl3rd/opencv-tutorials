#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// The scale of the drawings.
//
#define SCALE 400

// Draw an ellipse in image at angle in degrees.
//
static void drawBlueEllipse(cv::Mat &image,
                            const cv::Point &center, double angle)
{
    static const cv::Scalar colorBlue(255, 0, 0);
    static const int thickness = 2;
    static const int lineType = 8;              // the default
    const cv::Size axes(SCALE / 4, SCALE / 16);
    const double beginAngle = 0;
    const double endAngle = 360;
    cv::ellipse(image, center, axes,
                angle, beginAngle, endAngle,
                colorBlue, thickness, lineType);
}

// Draw a filled circle in image at center.
//
static void drawRedFilledCircle(cv::Mat &image,
                                const cv::Point &center, int radius)
{
    static const cv::Scalar colorRed(0, 0, 255);
    static const int thickness = -1;
    static const int lineType = 8;
    cv::circle(image, center, radius, colorRed, thickness, lineType);
}

// Draw a rook as a filled polygon.
//
void drawWhiteRookFilledPolygon(cv::Mat &image)
{
    static const cv::Scalar colorWhite(255, 255, 255);
    static const int lineType = 8;
    cv::Point points[1][20];
    points[0][ 0] = cv::Point(      SCALE /  4,   7 * SCALE /  8 );
    points[0][ 1] = cv::Point(  3 * SCALE /  4,   7 * SCALE /  8 );
    points[0][ 2] = cv::Point(  3 * SCALE /  4,  13 * SCALE / 16 );
    points[0][ 3] = cv::Point( 11 * SCALE / 16,  13 * SCALE / 16 );
    points[0][ 4] = cv::Point( 19 * SCALE / 32,   3 * SCALE /  8 );
    points[0][ 5] = cv::Point(  3 * SCALE /  4,   3 * SCALE /  8 );
    points[0][ 6] = cv::Point(  3 * SCALE /  4,       SCALE /  8 );
    points[0][ 7] = cv::Point( 26 * SCALE / 40,       SCALE /  8 );
    points[0][ 8] = cv::Point( 26 * SCALE / 40,       SCALE /  4 );
    points[0][ 9] = cv::Point( 22 * SCALE / 40,       SCALE /  4 );
    points[0][10] = cv::Point( 22 * SCALE / 40,       SCALE /  8 );
    points[0][11] = cv::Point( 18 * SCALE / 40,       SCALE /  8 );
    points[0][12] = cv::Point( 18 * SCALE / 40,       SCALE /  4 );
    points[0][13] = cv::Point( 14 * SCALE / 40,       SCALE /  4 );
    points[0][14] = cv::Point( 14 * SCALE / 40,       SCALE /  8 );
    points[0][15] = cv::Point(      SCALE /  4,       SCALE /  8 );
    points[0][16] = cv::Point(      SCALE /  4,   3 * SCALE /  8 );
    points[0][17] = cv::Point( 13 * SCALE / 32,   3 * SCALE /  8 );
    points[0][18] = cv::Point(  5 * SCALE / 16,  13 * SCALE / 16 );
    points[0][19] = cv::Point(      SCALE /  4,  13 * SCALE / 16 );
    const cv::Point *ppt[1] = { points[0] };
    const int npt[] = { 20 };
    static const int nContours = 1;
    cv::fillPoly(image, ppt, npt, nContours, colorWhite, lineType);
}

// Draw a rectangle on image.
//
static void drawYellowRectangle(cv::Mat &image)
{
    static const cv::Point vertex1(     0, 7 * SCALE / 8 );
    static const cv::Point vertex2( SCALE,     SCALE     );
    static const cv::Scalar colorYellow(0, 255, 255);
    static const int thickness = -1;
    static const int lineType = 8;
    cv::rectangle(image, vertex1, vertex2, colorYellow, thickness, lineType);
}

// Draw a line from begin to end on image.
//
static void drawBlackLine(cv::Mat &image,
                          const cv::Point &begin,
                          const cv::Point &end)
{
    static const cv::Scalar colorBlack(0, 0, 0);
    static const int thickness = 2;
    static const int lineType = 8;
    cv::line(image, begin, end, colorBlack, thickness, lineType);
}

// Draw some lines on image.
//
static void drawBlackLines(cv::Mat &image)
{
    static const struct Line {
        const cv::Point begin;
        const cv::Point end;
        Line(cv::Point b, cv::Point e): begin(b), end(e) {}
    } ln[] = {
        Line(cv::Point(             0, 15 * SCALE / 16 ) ,
             cv::Point(     SCALE    , 15 * SCALE / 16 )),
        Line(cv::Point(     SCALE / 4,  7 * SCALE /  8 ) ,
             cv::Point(     SCALE / 4,      SCALE      )),
        Line(cv::Point(     SCALE / 2,  7 * SCALE /  8 ) ,
             cv::Point(     SCALE / 2,      SCALE      )),
        Line(cv::Point( 3 * SCALE / 4,  7 * SCALE /  8 ) ,
             cv::Point( 3 * SCALE / 4,      SCALE      ))
    };
    static const int n = sizeof ln / sizeof ln[0];
    for (int i = 0; i < n; ++i) drawBlackLine(image, ln[i].begin, ln[i].end);
}

int main(int ac, const char *av[])
{
    static const cv::Point center(SCALE / 2, SCALE / 2);
    static const int radius = SCALE / 32;
    cv::Mat atomImage = cv::Mat::zeros(SCALE, SCALE, CV_8UC3);
    cv::Mat rookImage = cv::Mat::zeros(SCALE, SCALE, CV_8UC3);
    drawBlueEllipse(atomImage, center, +90);
    drawBlueEllipse(atomImage, center, +00);
    drawBlueEllipse(atomImage, center, +45);
    drawBlueEllipse(atomImage, center, -45);
    drawRedFilledCircle(atomImage, center, radius);
    drawWhiteRookFilledPolygon(rookImage);
    drawYellowRectangle(rookImage);
    drawBlackLines(rookImage);
    cv::imshow("Drawing 1: Atom", atomImage);
    cv::moveWindow("Drawing 1: Atom", 0, 200);
    cv::imshow("Drawing 2: Rook", rookImage);
    cv::moveWindow("Drawing 2: Rook", SCALE, 200);
    cv::waitKey(0);
    return 0;
}
