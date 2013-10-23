#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ts/ts_perf.hpp>
#include <iostream>
#include <sstream>

// Show a usage message on cout for program named av0.
//
static void showUsage(const char *av0)
{
    std::cout
        << av0 << ": Time scanning a Mat with the C operator[] method, "
        << std::endl
        << "    matrix iterators, the at() function, and the LUT() function."
        << std::endl << std::endl
        << "Usage: " << av0 << " <image-file> <divisor> [G]"
        << std::endl << std::endl
        << "Where: <image-file> is the path to an image file."
        << std::endl
        << "       The image should have a Mat::depth() of CV_8U."
        << std::endl
        << "       <divisor> is a small integer less than 255."
        << std::endl
        << "       G means process the image in gray scale."
        << std::endl << std::endl
        << "Example: " << av0 << " ../resources/Twas_Ever_Thus500.jpg 10"
        << std::endl
        << "Read an image object from Twas_Ever_Thus500 into a cv::Mat."
        << std::endl
        << "Repeatedly divide the image's native color palette by 10."
        << std::endl << std::endl;
}

// Return divisor after loading an image file into img.
// Return 0 after showing a usage message if there's a problem.
//
static int useCommandLine(int ac, const char *av[], cv::Mat &img)
{
    if (ac > 2) {
        int divisor = 0;
        std::stringstream ss; ss << av[2]; ss >> divisor;
        if (ss && divisor) {
            const bool g = (ac == 4) && (*"g" == *av[3]);
            const int cogOpt = g? CV_LOAD_IMAGE_GRAYSCALE: CV_LOAD_IMAGE_COLOR;
            img = cv::imread(av[1], cogOpt);
            if (img.data) return divisor;
        }
    }
    showUsage(av[0]);
    return 0;
}


// Call scan(*this) a lot and report the average run time in milliseconds.
//
// After cloning image, scan() visits every element, reduces it according
// to table or lookupTable, and returns the result.
//
// Each test() runs scan() many times and reports its average run time.
//
typedef struct Test {
    const cv::Mat &table;
    const cv::Mat &image;
    const char *const label;
    cv::Mat (*scan)(const struct Test &);

    void operator()(void) const {
        static const int runs = 100;
        const double tickZero = cv::getTickCount();
        for (int i = 0; i < runs; ++i) const cv::Mat ignore = (*scan)(*this);
        const double ticks = cv::getTickCount() - tickZero;
        const double totalSeconds = ticks / cv::getTickFrequency();
        const double msPerRun = totalSeconds * 1000 / runs;
        std::cout << "Average time to reduce with " << label << ": "
                  << msPerRun << " milliseconds." << std::endl;
    }

    Test(const cv::Mat &lut, const cv::Mat &i, const char *m,
         cv::Mat (*s)(const struct Test &)):
        table(lut), image(i), label(m), scan(s) {}
} Test;


static cv::Mat scanWithArrayOp(const Test &t)
{
    CV_Assert(CV_8U == t.image.depth());
    cv::Mat image = t.image.clone();
    int nRows = image.rows;
    int nCols = image.cols * image.channels();
    if (image.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }
    const uchar *const table = t.table.data;
    for (int i = 0; i < nRows; ++i) {
        uchar *const p = image.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j) p[j] = table[p[j]];
    }
    return image;
}


static cv::Mat scanWithMatIter(const Test &t)
{
    CV_Assert(CV_8U == t.image.depth());
    cv::Mat image = t.image.clone();
    const uchar *const table = t.table.data;
    switch (image.channels()) {
    case 1: {
        cv::MatIterator_<uchar> it = image.begin<uchar>();
        const cv::MatIterator_<uchar> end = image.end<uchar>();
        for ( ; it != end; ++it) *it = table[*it];
        break;
    }
    case 3: {
        cv::MatIterator_<cv::Vec3b> it = image.begin<cv::Vec3b>();
        const cv::MatIterator_<cv::Vec3b> end = image.end<cv::Vec3b>();
        for( ; it != end; ++it) {
            (*it)[0] = table[(*it)[0]];
            (*it)[1] = table[(*it)[1]];
            (*it)[2] = table[(*it)[2]];
        }
    }
    }
    return image;
}


static cv::Mat scanWithAt(const Test &t)
{
    CV_Assert(CV_8U == t.image.depth());
    cv::Mat image = t.image.clone();
    const uchar *const table = t.table.data;
    switch (image.channels()) {
    case 1: {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                image.at<uchar>(i,j) = table[image.at<uchar>(i,j)];
            }
        }
        break;
    }
    case 3: {
        cv::Mat_<cv::Vec3b> _I = image;
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                _I(i,j)[0] = table[_I(i,j)[0]];
                _I(i,j)[1] = table[_I(i,j)[1]];
                _I(i,j)[2] = table[_I(i,j)[2]];
            }
        }
        image = _I;
        break;
    }
    }
    return image;
}


static cv::Mat scanWithLut(const Test &t)
{
    CV_Assert(CV_8U == t.image.depth());
    cv::Mat J = t.image.clone();
    LUT(t.image, t.table, J);
    return J;
}


int main(int ac, const char *av[])
{
    cv::Mat image;
    const int divisor = useCommandLine(ac, av, image);
    if (divisor == 0) return 1;
    cv::Mat table(1, 256, CV_8U);
    uchar *const p = table.data;
    for (int i = 0; i < table.cols; ++i) p[i] = (divisor * (i / divisor));
    const Test tests[] = {
        Test(table, image, "operator[]", &scanWithArrayOp),
        Test(table, image, "iterator",   &scanWithMatIter),
        Test(table, image, "at()",       &scanWithAt),
        Test(table, image, "LUT()",      &scanWithLut)
    };
    const int testsCount = sizeof tests / sizeof tests[0];
    for (int i = 0; i < testsCount; ++i) (tests[i])();
    return 0;
}
