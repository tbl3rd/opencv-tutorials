#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


#define ARRAY_COUNT(A) ((sizeof (A)) / (sizeof ((A)[0])))

static struct ThresholdKind {
    const char *name;
    int value;
    ThresholdKind(): name(0), value(0) {}
    ThresholdKind(const char *n, int v): name(n), value(v) {}
} theThresholdKinds[] = {
    ThresholdKind("binary",           cv::THRESH_BINARY    ),
    ThresholdKind("binary inverted",  cv::THRESH_BINARY_INV),
    ThresholdKind("truncated",        cv::THRESH_TRUNC     ),
    ThresholdKind("to zero",          cv::THRESH_TOZERO    ),
    ThresholdKind("to zero inverted", cv::THRESH_TOZERO_INV)
};
static const int theThresholdKindCount = ARRAY_COUNT(theThresholdKinds);

// The max limit of the threshold.
//
static const int theMaxValue = std::numeric_limits<uchar>::max();


// A display to demonstrate basic thresholding.
//
class DemoDisplay {

protected:

    // The source and destination images for apply().
    //
    const cv::Mat &srcImage;
    cv::Mat dstImage;

    // Apply threshold of kind with value below max to srcImage producing
    // dstImage.
    //
    void apply(double value, double max, int kind)
    {
        cv::threshold(srcImage, dstImage, value, max, kind);
    }

private:

    // The window caption.
    //
    const char *const caption;

    // The position of the Threshold Kind and Threshold Value trackbars.
    //
    int kindBar;
    int valueBar;

    // The callback passed to createTrackbar() where all state is at p.
    //
    static void show(int positionIgnoredUseThisInstead,  void *p)
    {
        static ThresholdKind oldKind;
        DemoDisplay *const pD = (DemoDisplay *)p;
        const double value = pD->valueBar;
        const double max = theMaxValue;
        const ThresholdKind kind = theThresholdKinds[pD->kindBar];
        if (!oldKind.name || kind.value != oldKind.value) {
            std::cout << "Threshold " << kind.value << ": " << kind.name
                      << std::endl;
            oldKind = kind;
        }
        pD->apply(value, max, kind.value);
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
        caption("Threshold Demo"), srcImage(s), kindBar(0), valueBar(0)
    {
        static const int maxKind = theThresholdKindCount - 1;
        cv::namedWindow(caption, cv::WINDOW_AUTOSIZE);
        makeTrackbar("Threshold Kind:",  &kindBar,  maxKind);
        makeTrackbar("Threshold Value:", &valueBar, theMaxValue);
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
            DemoDisplay demo(gray); demo();
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate some basic thresholding."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file." 
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/chicky_512.png"
              << std::endl << std::endl;
    return 1;
}
