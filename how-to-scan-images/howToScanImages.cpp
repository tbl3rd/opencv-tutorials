#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

// Show a usage message on cout for program named av0.
//
static void showUsage(const char *av0)
{
    std::cout
        << "Scan image objects in OpenCV (cv::Mat)."
        << std::endl
        << "As use case take an input image and divide"
        << std::endl
        << "the native color palette (255) with the input."
        << std::endl
        << "Shows C operator[] method, iterators, and at() function"
        << std::endl
        << "for on-the-fly item address calculation."
        << std::endl
        << "Usage: " << av0 << " imageNameToUse divideWith [G]"
        << std::endl
        << "If you add a G parameter the image is processed in gray scale"
        << std::endl << std::endl;
}

// Return divideWith after loading imageNameToUse into img.
// Return 0 after showing a usage message if there's a problem.
//
static int useCommandLine(int ac, const char *av[], cv::Mat &img)
{
    if (ac > 2) {
        int divideWith = 0;
        std::stringstream ss; ss << av[2]; ss >> divideWith;
        if (ss && divideWith) {
            const bool g = (ac == 4) && (*"g" == *av[3]);
            const int cogOpt = g? CV_LOAD_IMAGE_GRAYSCALE: CV_LOAD_IMAGE_COLOR;
            img = cv::imread(av[1], cogOpt);
            if (img.data) return divideWith;
        }
    }
    showUsage(av[0]);
    return 0;
}


static cv::Mat &ScanImageAndReduceArrayOp(cv::Mat &I, const uchar table[])
{
    CV_Assert(I.depth() != sizeof(uchar));
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
    CV_Assert(I.depth() != sizeof(uchar));
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
    CV_Assert(I.depth() != sizeof(uchar));
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
        cv::Mat clone_i = I.clone();
        cv::Mat J = ScanImageAndReduceArrayOp(clone_i, table);
    }
    t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t /= times;
    return t;
}


static double reduceWithMatrixIterator(const cv::Mat &I, const uchar table[])
{
    static const int times = 100;
    double t = cv::getTickCount();
    for (int i = 0; i < times; ++i) {
        cv::Mat clone_i = I.clone();
        cv::Mat J = ScanImageAndReduceIterator(clone_i, table);
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
        cv::Mat clone_i = I.clone();
        cv::Mat J = ScanImageAndReduceRandom(clone_i, table);
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
    for (int i = 0; i < times; ++i) LUT(I, lookUpTable, J);
    t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    t /= times;
    return t;
}


int main(int ac, const char *av[])
{
    cv::Mat I, J;
    const int divideWith = useCommandLine(ac, av, I);
    if (divideWith == 0) return 1;
    uchar table[256];
    const int tableSize = sizeof table / sizeof table[0];
    for (int i = 0; i < tableSize; ++i) {
        table[i] = (divideWith * (i / divideWith));
    }
    double t = reduceWithArrayOp(I, table);
    std::cout << "Average time to reduce with operator[]:  "
              << t << " milliseconds."<< std::endl;
    t = reduceWithMatrixIterator(I, table);
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
