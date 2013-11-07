#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Create a new unobscured named window for image.
//
// The fudge works around how MacOSX lays out window decorations.
//
static void makeWindow(const char *window, const cv::Mat &image,
                       bool reset = false)
{
    static const int across = 2;
    static int moveCount = 0;
    if (reset) moveCount = 0;
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    const int overCount = moveCount % across;
    const int downCount = moveCount / across;
    const int moveX = overCount * image.cols;
    const int moveY = downCount * image.rows;
    const int fudge = downCount == 0 ? 0 : (1 + downCount);
    cv::moveWindow(window, moveX, moveY + 23 * fudge);
    ++moveCount;
}


// An map m mamed n for an image of size s that computes
//
//    dst(x, y) = src(m.itsX(x, y), m.itsY(x, y))
//
// where ImageMap::init(m) calls m.computeXandYmaps(i, j) once for all rows
// i and all columns j to initialize m.itsX and m.itsY.
//
class ImageMap {

    const char *itsName;

    virtual void computeXandYmaps(int i, int j) = 0;

protected:

    cv::Mat itsX;
    cv::Mat itsY;

    static void init(ImageMap &m)
    {
        assert(m.itsX.size() == m.itsY.size());
        const int rows = m.itsX.rows;
        const int cols = m.itsX.cols;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                m.computeXandYmaps(i, j);
            }
        }
    }

public:

    virtual ~ImageMap() {}

    cv::Mat operator()(const cv::Mat &image) const
    {
        static const int interpolation = cv::INTER_LINEAR;
        static const int borderKind = cv::BORDER_CONSTANT;
        static const cv::Scalar borderValue(0, 0, 0);
        cv::Mat result;
        cv::remap(image, result, itsX, itsY,
                  interpolation, borderKind, borderValue);
        return result;
    }

    const char *name() const { return itsName; }

    ImageMap(const char *n, const cv::Size &s):
        itsName(n),
        itsX(s, CV_32FC1),
        itsY(s, CV_32FC1)
    {}
};


// The identity map leaves the image unchanged.
//
class IdentityMap: public ImageMap {
    virtual void computeXandYmaps(int i, int j)
    {
        itsX.at<float>(i, j) = j;
        itsY.at<float>(i, j) = i;
    }
public:
    IdentityMap(const cv::Size &size):
        ImageMap("Identity", size)
    {
        ImageMap::init(*this);
    }
};

// Reflect image around its horizontal axis.
//
class ReflectHorizontalMap: public ImageMap {
    virtual void computeXandYmaps(int i, int j)
    {
        itsX.at<float>(i, j) = j;
        itsY.at<float>(i, j) = itsY.rows - i;
    }
public:
    ReflectHorizontalMap(const cv::Size &size):
        ImageMap("Reflect Horizontal", size)
    {
        ImageMap::init(*this);
    }
};

// Reflect image around its vertical axis.
//
class ReflectVerticalMap: public ImageMap {
    virtual void computeXandYmaps(int i, int j)
    {
        itsX.at<float>(i, j) = itsX.cols - j;
        itsY.at<float>(i, j) = i;
    }
public:
    ReflectVerticalMap(const cv::Size &size):
        ImageMap("Reflect Vertical", size)
    {
        ImageMap::init(*this);
    }
};

// Reflect image around both the horizontal and vertical axes --
// effectively rotating it 180 degrees about its center point.
//
class ReflectHorizontalVerticalMap: public ImageMap {
    virtual void computeXandYmaps(int i, int j)
    {
        itsX.at<float>(i, j) = itsX.cols - j;
        itsY.at<float>(i, j) = itsY.rows - i;
    }
public:
    ReflectHorizontalVerticalMap(const cv::Size &size):
        ImageMap("Reflect Horizontal Vertical", size)
    {
        ImageMap::init(*this);
    }
};

// Center image at half scale.
//
class HalfScaleMap: public ImageMap {
    virtual void computeXandYmaps(int i, int j)
    {
        const int minCols = itsX.cols / 4;
        const int maxCols = 3 * minCols;
        const int minRows = itsY.rows / 4;
        const int maxRows = 3 * minRows;
        float x = 0.0;
        float y = 0.0;
        const bool ok
            =  i < maxRows && i > minRows
            && j < maxCols && j > minCols;
        if (ok) {
            x = 0.5 + 2 * (j - minCols);
            y = 0.5 + 2 * (i - minRows);
        }
        itsX.at<float>(i, j) = x;
        itsY.at<float>(i, j) = y;
    }
public:
    HalfScaleMap(const cv::Size &size):
        ImageMap("Half Scale", size)
    {
        ImageMap::init(*this);
    }
};

// Show the result of applying map to src.
//
static void showMap(const ImageMap &map, const cv::Mat &src)
{
    const cv::Mat dst = map(src);
    makeWindow(map.name(), dst);
    cv::imshow(map.name(), dst);
}

// Show a remap of src in window by cycling through the mapCount image maps
// in maps once per second until the user keys 'q'.
//
static void showRemaps(const char *window, const cv::Mat &src,
                       int mapCount, const ImageMap *maps[])
{
    static const int oneSecondInMilliseconds = 1000;
    int index = 0;
    while (true) {
        const ImageMap &map = *maps[index %= mapCount];
        const cv::Mat dst = map(src);
        cv::imshow(window, dst);
        const int c = cv::waitKey(oneSecondInMilliseconds);
        if ('Q' == c || 'q' == c) break;
        ++index;
    }
}

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat src = cv::imread(av[1]);
        if (src.data) {
            std::cout << av[0] << ": Press 'q' to quit." << std::endl;
            makeWindow("Remap demo", src, true);
            const cv::Size size = src.size();
            const IdentityMap                  id(size);
            const ReflectHorizontalMap         rh(size);
            const ReflectVerticalMap           rv(size);
            const ReflectHorizontalVerticalMap rb(size);
            const HalfScaleMap                 qs(size);
            const ImageMap *map[] = { &id, &rh, &rv, &rb, &qs };
            const int mapCount = sizeof map / sizeof map[0];
            for (int i = 0; i < mapCount; ++i) showMap(*map[i], src);
            showRemaps("Remap demo", src, mapCount, map);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate image remapping."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
