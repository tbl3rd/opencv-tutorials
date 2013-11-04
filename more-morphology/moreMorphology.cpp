#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


#define ARRAY_COUNT(A) ((sizeof (A)) / (sizeof ((A)[0])))

// The morphology operators supported by getStructuringElement().
//
static const int theMorphOps[] = {
    cv::MORPH_OPEN,     cv::MORPH_CLOSE,
    cv::MORPH_GRADIENT, cv::MORPH_TOPHAT, cv::MORPH_BLACKHAT
};
static const int theMorphOpsCount = ARRAY_COUNT(theMorphOps);

// The element shapes supported by getStructuringElement().
//
static const int theElementShapes[] = {
    cv::MORPH_RECT, cv::MORPH_CROSS, cv::MORPH_ELLIPSE,
};
static const int theElementShapesCount = ARRAY_COUNT(theElementShapes);


// A display to demonstrate some morphology operators.
//
class DemoDisplay {

protected:

    // The source and destination images for apply().
    //
    const cv::Mat &srcImage;
    cv::Mat dstImage;

    // Apply operation with element to srcImage producing dstImage.
    //
    void apply(int operation, const cv::Mat &element)
    {
        cv::morphologyEx(srcImage, dstImage, operation, element);
    }

private:

    // The window caption.
    //
    const char *const caption;

    // The position of the Morphology Operation, Element Shape,
    // and Kernel Shape trackbars.
    //
    int opBar;
    int elementBar;
    int sizeBar;

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnoredUseThisInstead,  void *p)
    {
        DemoDisplay *pD = (DemoDisplay *)p;
        assert(pD->elementBar < theElementShapesCount);
        assert(pD->opBar < theMorphOpsCount);
        const int operation = theMorphOps[pD->opBar];
        const int shape = theElementShapes[pD->elementBar];
        const int size = 1 + 2 * pD->sizeBar;
        const cv::Size kernelSize(size, size);
        const cv::Point anchor(pD->sizeBar, pD->sizeBar);
        const cv::Mat element
            = cv::getStructuringElement(shape, kernelSize, anchor);
        pD->apply(operation, element);
        cv::imshow(pD->caption, pD->dstImage);
    }

    // Add a trackbar with label of range 0 to max in bar.
    //
    void makeTrackbar(const char *label, int *bar, int max)
    {
        cv::createTrackbar(label, caption, bar, max, &show, this);
    }

public:

    // Show this demo display window.
    //
    void operator()(void) { DemoDisplay::show(0, this); }

    // Construct a display with the caption c operating on source image s.
    //
    DemoDisplay(const cv::Mat &s):
        caption("Morphology Transformations Demo"), srcImage(s),
        opBar(0), elementBar(0), sizeBar(0)
    {
        static const int maxOp = theMorphOpsCount - 1;
        static const int maxElement = theElementShapesCount - 1;
        static const int maxKernelSize = 21;
        static int moveX = 0;
        cv::namedWindow(caption, cv::WINDOW_AUTOSIZE);
        makeTrackbar("Morph Operator:", &opBar,      maxOp);
        makeTrackbar("Element Shape:",  &elementBar, maxElement);
        makeTrackbar("Kernel Size:",    &sizeBar,    maxKernelSize);
        cv::moveWindow(caption, moveX, 0);
        moveX += srcImage.cols;
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat srcImage = cv::imread(av[1]);
        if (srcImage.data) {
            DemoDisplay demo(srcImage); demo();
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate some more morphology operations."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file." 
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/mandrill.tiff"
              << std::endl << std::endl;
    return 1;
}
