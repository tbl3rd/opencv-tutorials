#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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


static cv::Mat &ScanImageAndReduceArrayOp(cv::Mat &I, const uchar table[])
{
    CV_Assert(CV_8U == I.depth());
    int nRows = I.rows;
    int nCols = I.cols * I.channels();
    if (I.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }
    for (int i = 0; i < nRows; ++i) {
        uchar *const p = I.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j) p[j] = table[p[j]];
    }
    return I;
}


static cv::Mat &ScanImageAndReduceIterator(cv::Mat &I, const uchar table[])
{
    CV_Assert(CV_8U == I.depth());
    switch (I.channels()) {
    case 1: {
        cv::MatIterator_<uchar> it = I.begin<uchar>();
        const cv::MatIterator_<uchar> end = I.end<uchar>();
        for ( ; it != end; ++it) *it = table[*it];
        break;
    }
    case 3: {
        cv::MatIterator_<cv::Vec3b> it = I.begin<cv::Vec3b>();
        const cv::MatIterator_<cv::Vec3b> end = I.end<cv::Vec3b>();
        for( ; it != end; ++it) {
            (*it)[0] = table[(*it)[0]];
            (*it)[1] = table[(*it)[1]];
            (*it)[2] = table[(*it)[2]];
        }
    }
    }
    return I;
}


static cv::Mat &ScanImageAndReduceRandom(cv::Mat &I, const uchar *const table)
{
    CV_Assert(CV_8U == I.depth());
    switch (I.channels()) {
    case 1: {
        for (int i = 0; i < I.rows; ++i) {
            for (int j = 0; j < I.cols; ++j) {
                I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];
            }
        }
        break;
    }
    case 3: {
        cv::Mat_<cv::Vec3b> _I = I;
        for (int i = 0; i < I.rows; ++i) {
            for (int j = 0; j < I.cols; ++j) {
                _I(i,j)[0] = table[_I(i,j)[0]];
                _I(i,j)[1] = table[_I(i,j)[1]];
                _I(i,j)[2] = table[_I(i,j)[2]];
            }
        }
        I = _I;
        break;
    }
    }
    return I;
}


static double reduceWithArrayOp(const cv::Mat &I, const uchar table[])
{
    static const int times = 100;
    double t = cv::getTickCount();
    for (int i = 0; i < times; ++i) {
        cv::Mat J = I.clone();
        J = ScanImageAndReduceArrayOp(J, table);
    }
    t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t /= times;
    return t;
}


static double reduceWithMatIter(const cv::Mat &I, const uchar table[])
{
    static const int times = 100;
    double t = cv::getTickCount();
    for (int i = 0; i < times; ++i) {
        cv::Mat J = I.clone();
        J = ScanImageAndReduceIterator(J, table);
    }
    t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t /= times;
    return t;
}


static double reduceWithAt(const cv::Mat &I, const uchar table[])
{
    static const int times = 100;
    double t = cv::getTickCount();
    for (int i = 0; i < times; ++i) {
        cv::Mat J = I.clone();
        J = ScanImageAndReduceRandom(J, table);
    }
    t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t /= times;
    return t;
}


static double reduceWithLookupTable(const cv::Mat &I, const uchar table[])

{
    static const int times = 100;
    cv::Mat J, lookUpTable(1, 256, CV_8U);
    uchar *const p = lookUpTable.data;
    for (int i = 0; i < 256; ++i) p[i] = table[i];
    double t = cv::getTickCount();
    for (int i = 0; i < times; ++i) {
        cv::Mat J = I.clone();
        LUT(I, lookUpTable, J);
    }
    t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t /= times;
    return t;
}


int main(int ac, const char *av[])
{
    cv::Mat I;
    const int divisor = useCommandLine(ac, av, I);
    if (divisor == 0) return 1;
    uchar table[256];
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar *const p = lookUpTable.data;
    const int tableSize = sizeof table / sizeof table[0];
    for (int i = 0; i < tableSize; ++i) {
        p[i] = table[i] = (divisor * (i / divisor));
    }
    double t = reduceWithArrayOp(I, table);
    std::cout << "Average time to reduce with operator[]:  "
              << t << " milliseconds."<< std::endl;
    t = reduceWithMatIter(I, table);
    std::cout << "Average time to reduce with MatIterator: "
              << t << " milliseconds."<< std::endl;
    t = reduceWithAt(I, table);
    std::cout << "Average time to reduce with at():        "
              << t << " milliseconds."<< std::endl;
    t = reduceWithLookupTable(I, table);
    std::cout << "Average time to reduce with LUT():       "
              << t << " milliseconds."<< std::endl;
    return 0;
}
