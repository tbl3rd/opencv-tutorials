#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
g
// Show a usage message on cout for program named av0.
//
static void showUsage(const char *av0)
{
    std::cout
        << av0 << ": Filter an image with a 'sharpening' mask."
        << std::endl << std::endl
        << "Usage: " << av0 << " <image-file> [g]"
        << std::endl << std::endl
        << "Where: <image-file> is the path to an image file."
        << std::endl
        << "       The image should have a Mat::depth() of CV_8U."
        << std::endl
        << "       g means process the image in gray scale."
        << std::endl << std::endl
        << "Example: " << av0 << " ../resources/lena.tiff"
        << std::endl
        << "Read an image object from lena.tiff into a cv::Mat."
        << std::endl
        << "Repeatedly sharpen the image by applying a mask cv::Mat."
        << std::endl << std::endl;
}

// Return the image specified on the command line.
// Show the usage message if something is wrong.
//
static cv::Mat useCommandLine(int ac, const char *av[])
{
    cv::Mat result;
    int ok = ac == 2 || ac == 3;
    if (ok) {
        const bool g = (ac == 3) && (*"g" == *av[2]);
        const int cogOpt = g? CV_LOAD_IMAGE_GRAYSCALE: CV_LOAD_IMAGE_COLOR;
        result = cv::imread(av[1], cogOpt);
        ok = CV_8U == result.depth();
    }
    if (!ok) showUsage(av[0]);
    return result;
}

// Run (*this)() to transform input to output according to label.
//
struct Test {
    const char *const label;
    const cv::Mat &input;
    cv::Mat output;
    virtual void operator()(void) = 0;
    Test(const char *l, const cv::Mat &i):
        label(l), input(i), output(i.size(), i.type())
    {}
};

// Apply the kernel mask to input and write result to output.
//
struct Filter2dTest: Test {
    const cv::Mat mask;
    void operator()(void) {
        cv::filter2D(input, output, input.depth(), mask);
    }
    static cv::Mat makeMask(void) {
        cv::Mat_<char> result(3,3);
        result <<
            +0, -1, +0,
            -1, +5, -1,
            +0, -1, +0;
        return result;
    }
    Filter2dTest(const cv::Mat &i): Test("filter2D()", i), mask(makeMask()) {}
};

// Do what Filter2dTest does via a hand-coded scanner.
//
struct HandCodedTest: Test {
    void operator()(void) {
        const int nChannels = input.channels();
        const int rowMax = input.rows - 1;
        const int colMax = input.cols - 1;
        for (int j = 1 ; j < rowMax; ++j) {
            const uchar *const previous = input.ptr<uchar>(j - 1);
            const uchar *const current  = input.ptr<uchar>(j    );
            const uchar *const next     = input.ptr<uchar>(j + 1);
            uchar *p = output.ptr<uchar>(j);
            for (int i = nChannels; i < nChannels * colMax; ++i) {
                const int sharper = 0           // 0 except for
                    - previous[i]               // [i    , j - 1] == -1
                    - current[i - nChannels]    // [i - 1, j    ] == -1
                    + 5 * current[i]            // [i    , j    ] == +5
                    - current[i + nChannels]    // [i + 1, j    ] == -1
                    - next[i];                  // [i    , j + 1] == -1
                *p++ = cv::saturate_cast<uchar>(sharper);
            }
        }
        static const cv::Scalar zero(0);        // Mask the border to 0.
        output.row(0).setTo(zero);              // Set first row to 0.
        output.row(rowMax).setTo(zero);         // Set last  row to 0.
        output.col(0).setTo(zero);              // Set first col to 0.
        output.col(colMax).setTo(zero);         // Set last  col to 0.
    }
    HandCodedTest(const cv::Mat &i): Test("hand-coded", i) {}
};

int main(int ac, const char *av[])
{
    const cv::Mat inputImage = useCommandLine(ac, av);
    if (!inputImage.data) return 1;
    cv::imshow(av[1], inputImage);
    HandCodedTest handCodedTest(inputImage);
    Filter2dTest builtinTest(inputImage);
    Test *tests[] = { &handCodedTest, &builtinTest };
    const int testCount = sizeof tests / sizeof tests[0];
    for (int i = 0; i < testCount; ++i) {
        Test &test = *tests[i];
        static const int runCount = 100;
        const int64 tickZero = cv::getTickCount();
        for (int j = 0; j < runCount; ++j) (test)();
        const int64 ticks = cv::getTickCount() - tickZero;
        const double totalSeconds = (double)ticks / cv::getTickFrequency();
        const double msPerRun = totalSeconds * 1000 / runCount;
        std::cout << "Average " << test.label << " time in milliseconds: "
                  << msPerRun << std::endl;
        cv::imshow(test.label, test.output);
    }
    cv::waitKey(0);
    return 0;
}
