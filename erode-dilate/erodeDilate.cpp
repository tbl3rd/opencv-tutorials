#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// A display to demonstrate the erode() and dilate() operators.
//
class Display {

    // The window caption.
    //
    const char *const caption;

    // The position of the Element Shape and Kernel Shape trackbars.
    //
    int elementBar;
    int sizeBar;

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnored,  void *p)
    {
        static const int elementShape[] = {
            cv::MORPH_RECT, cv::MORPH_CROSS, cv::MORPH_ELLIPSE,
        };
        static const int count = sizeof elementShape / sizeof elementShape[0];
        Display *pD = (Display *)p;
        assert(pD->elementBar < count);
        const int shape = elementShape[pD->elementBar];
        const int size = 1 + 2 * pD->sizeBar;
        const cv::Size kernelSize(size, size);
        const cv::Point anchor(pD->sizeBar, pD->sizeBar);
        const cv::Mat element
            = cv::getStructuringElement(shape, kernelSize, anchor);
        pD->erodeOrDilate(element);
        cv::imshow(pD->caption, pD->dstImage);
    }

protected:

    // The source and destination images for erodeOrDilate().
    //
    const cv::Mat &srcImage;
    cv::Mat dstImage;

    // Override this to call f(srcImage, dstImage, element) where f() is
    // either erode() or dilate().
    //
    virtual void erodeOrDilate(const cv::Mat &element) = 0;

public:

    // Show this demo display.
    //
    void operator()(void) { Display::show(0, this); }

    // Construct a display with the caption c operating on source image s.
    //
    Display(const char *c, const cv::Mat &s):
        caption(c), srcImage(s), elementBar(0), sizeBar(0)
    {
        static const int maxElement = 2;
        static const int maxKernelSize = 21;
        static int moveX = 0;
        cv::namedWindow(caption, cv::WINDOW_AUTOSIZE);
        cv::moveWindow(caption, moveX, 0);
        cv::createTrackbar("Element Shape:",
                           caption, &elementBar, maxElement, &show, this);
        cv::createTrackbar("Kernel Size:",
                           caption, &sizeBar, maxKernelSize, &show, this);
        moveX += srcImage.cols;
    }
};

// Call erode() from Display::show().
//
class ErosionDisplay: public Display {
    virtual void erodeOrDilate(const cv::Mat &element)
    {
        cv::erode(srcImage, dstImage, element);
    }
public:
    ErosionDisplay(const cv::Mat &s): Display("Erosion Demo", s) {}
};

// Call dilate() from Display::show().
//
class DilationDisplay: public Display {
    virtual void erodeOrDilate(const cv::Mat &element)
    {
        cv::dilate(srcImage, dstImage, element);
    }
public:
    DilationDisplay(const cv::Mat &s): Display("Dilation Demo", s) {}
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat srcImage = cv::imread(av[1]);
        if (srcImage.data) {
            ErosionDisplay  erode(srcImage);  erode();
            DilationDisplay dilate(srcImage); dilate();
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
