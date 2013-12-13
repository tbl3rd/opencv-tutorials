#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// The element shapes supported by getStructuringElement().
//
static const int theElementShapes[] = {
    cv::MORPH_RECT, cv::MORPH_CROSS, cv::MORPH_ELLIPSE,
};
static const int theElementShapesCount =
    sizeof theElementShapes / sizeof theElementShapes[0];


// A display to demonstrate the erode() and dilate() morphology operators.
//
class DemoDisplay {

protected:

    // The source and destination images for apply().
    //
    const cv::Mat &srcImage;
    cv::Mat dstImage;

    // Override this to call f(srcImage, dstImage, element) where f() is
    // either erode() or dilate().
    //
    virtual void apply(const cv::Mat &element) = 0;

private:

    // The window caption.
    //
    const char *const caption;

    // The position of the Element Shape and Kernel Shape trackbars.
    //
    int elementBar;
    int sizeBar;

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnoredUseThisInstead,  void *p)
    {
        DemoDisplay *const pD = (DemoDisplay *)p;
        const int shape = theElementShapes[pD->elementBar];
        const int size = 1 + 2 * pD->sizeBar;
        const cv::Size kernelSize(size, size);
        const cv::Point anchor(pD->sizeBar, pD->sizeBar);
        const cv::Mat element
            = cv::getStructuringElement(shape, kernelSize, anchor);
        pD->apply(element);
        cv::imshow(pD->caption, pD->dstImage);
    }

    // Add a trackbar with label of range 0 to max in bar.
    //
    void makeTrackbar(const char *label, int *bar, int max)
    {
        cv::createTrackbar(label, caption, bar, max, &show, this);
    }

public:

    virtual ~DemoDisplay() {}

    // Show this demo display window.
    //
    void operator()(void) { DemoDisplay::show(0, this); }

    // Construct a display with the caption c operating on source image s.
    //
    DemoDisplay(const char *c, const cv::Mat &s):
        caption(c), srcImage(s), elementBar(0), sizeBar(0)
    {
        static const int maxElement = theElementShapesCount - 1;
        static const int maxKernelSize = 21;
        static int moveX = 0;
        cv::namedWindow(caption, cv::WINDOW_AUTOSIZE);
        makeTrackbar("Element Shape:", &elementBar, maxElement);
        makeTrackbar("Kernel Size:",   &sizeBar,    maxKernelSize);
        cv::moveWindow(caption, moveX, 0);
        moveX += srcImage.cols;
    }
};

// Call erode() from DemoDisplay::show().
//
class ErosionDemoDisplay: public DemoDisplay {
    virtual void apply(const cv::Mat &element)
    {
        cv::erode(srcImage, dstImage, element);
    }
public:
    ErosionDemoDisplay(const cv::Mat &s): DemoDisplay("Erosion Demo", s) {}
};

// Call dilate() from DemoDisplay::show().
//
class DilationDemoDisplay: public DemoDisplay {
    virtual void apply(const cv::Mat &element)
    {
        cv::dilate(srcImage, dstImage, element);
    }
public:
    DilationDemoDisplay(const cv::Mat &s): DemoDisplay("Dilation Demo", s) {}
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat srcImage = cv::imread(av[1]);
        if (srcImage.data) {
            ErosionDemoDisplay  erode(srcImage);  erode();
            DilationDemoDisplay dilate(srcImage); dilate();
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate erosion and dilation." << std::endl
              << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file." 
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
