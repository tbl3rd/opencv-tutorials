#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


static int elemToShape(int erosion_elem)
{
    static const int shape[] = {
        cv::MORPH_RECT,
        cv::MORPH_CROSS,
        cv::MORPH_ELLIPSE,
    };
    static const int count = sizeof shape / sizeof shape[0];
    assert(erosion_elem < count);
    return shape[erosion_elem];
}

typedef void (*TrackbarCallback)(int position, void *cbState);
typedef void (*ErodeOrDilate)(const cv::Mat &src,
                              cv::Mat &dst,
                              const cv::Mat &kernel);

struct Display {
    const char *const caption;
    TrackbarCallback fn;
    ErodeOrDilate something;
    const cv::Mat &src;
    cv::Mat dst;
    int elem;
    int size;
    Display(const char *c, TrackbarCallback f, const cv::Mat &s):
        caption(c), fn(f), src(s), elem(0), size(0)
    {}
};

static void showErosion(int, void *p)
{
    Display *pD = (Display *)p;
    const int shape = elemToShape(pD->elem);
    const int size = 1 + 2 * pD->size;
    const cv::Size kernelSize(size, size);
    const cv::Point anchor(pD->size, pD->size);
    cv::Mat element = cv::getStructuringElement(shape, kernelSize, anchor);
    cv::erode(pD->src, pD->dst, element);
    cv::imshow(pD->caption, pD->dst);
}

static void showDilation(int,  void *p)
{
    Display *pD = (Display *)p;
    const int shape = elemToShape(pD->elem);
    const int size = 1 + 2 * pD->size;
    const cv::Size kernelSize(size, size);
    const cv::Point anchor(pD->size, pD->size);
    cv::Mat element = cv::getStructuringElement(shape, kernelSize, anchor);
    cv::dilate(pD->src, pD->dst, element);
    cv::imshow(pD->caption, pD->dst);
}

static void showSomething(int,  void *p)
{
    Display *pD = (Display *)p;
    const int shape = elemToShape(pD->elem);
    const int size = 1 + 2 * pD->size;
    const cv::Size kernelSize(size, size);
    const cv::Point anchor(pD->size, pD->size);
    cv::Mat element = cv::getStructuringElement(shape, kernelSize, anchor);
    (*pD->something)(pD->src, pD->dst, element);
    cv::imshow(pD->caption, pD->dst);
}

static void makeTrackbarWindow(Display &d)
{
    static const int maxElem = 2;
    static const int maxKernelSize = 21;
    static int moveX = 0;
    cv::namedWindow(d.caption, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(d.caption, moveX, 0);
    cv::createTrackbar("Element Shape:",
                       d.caption, &d.elem, maxElem, d.fn, &d);
    cv::createTrackbar("Kernel Size:",
                       d.caption, &d.size, maxKernelSize, d.fn, &d);
    moveX += d.src.cols;
}


int main(int argc, const char *argv[])
{
    const cv::Mat src = cv::imread(argv[1]);
    if (!src.data) return -1;

    Display erosion("Erosion Demo", showErosion, src);
    Display dilation("Dilation Demo", showDilation, src);

    makeTrackbarWindow(erosion);
    makeTrackbarWindow(dilation);

    showErosion(0, &erosion);
    showDilation(0, &dilation);

    cv::waitKey(0);
    return 0;
}
