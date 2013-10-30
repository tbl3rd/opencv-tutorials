#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

static void showUsage(const char *av0)
{
    std::cout
        << std::endl
        << av0 << ": Demonstrate serializing data to and from files."
        << std::endl << std::endl
        << "Usage: " <<  av0 << " <file><ext>" << std::endl
        << std::endl
        << "Where: <file><ext> is the name of a file to read and write."
        << std::endl
        << "       The <ext> extension may be: '.xml' or '.yaml'" << std::endl
        << "       to serialize data as XML or as YAML, respectively."
        << std::endl << std::endl
        << "       A '.gz' suffix designates compression such that:"
        << std::endl
        << "           <file>.xml.gz  means use gzipped XML." << std::endl
        << "           <file>.yaml.gz means use gzipped YAML." << std::endl
        << std::endl
        << "Example: " << av0 << " somedata.xml.gz" << std::endl
        << std::endl;
}

struct SomeData
{
    int anInt;
    double aDouble;
    std::string aString;

    SomeData(): anInt(1), aDouble(1.1), aString("default one dot one") {}

    SomeData(int): anInt(97), aDouble(CV_PI), aString("mydata1234") {}

    void write(cv::FileStorage &fs) const
    {
        fs << "{"
           << "anInt"   << anInt
           << "aDouble" << aDouble
           << "aString" << aString
           << "}";
    }
    void read(const cv::FileNode &node)
    {
        anInt   = int(        node["anInt"  ]);
        aDouble = double(     node["aDouble"]);
        aString = std::string(node["aString"]);
    }
};

static void write(cv::FileStorage &fs, const std::string &, const SomeData &x)
{
    x.write(fs);
}
static void read(const cv::FileNode &node, SomeData &x,
                 const SomeData &default_value = SomeData())
{
    if (node.empty()) {
        x = default_value;
    } else {
        x.read(node);
    }
}

static std::ostream &operator<<(std::ostream &out, const SomeData &m)
{
    out << "anInt"   << "="         << m.anInt
        << ", "
        << "aDouble" << "="         << m.aDouble
        << ", "
        << "aString" << "=" << "\"" << m.aString << "\"";

    return out;
}

static bool writeSomeData(const char *filename)
{
    std::cout << "Writing ... ";
    cv::Mat ucharEye = cv::Mat_<uchar>::eye(3, 3);
    cv::Mat doubleZeros = cv::Mat_<double>::zeros(3, 1);
    SomeData someData(1);
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "iterationNr" << 100
       << "strings"
       << "[" << "image1.jpg" << "Awesomeness" << "baboon.jpg" << "]"
       << "mapping" << "{" << "One" << 1 << "Two" << 2 << "}"
       << "ucharEye" << ucharEye << "doubleZeros" << doubleZeros
       << "someData" << someData;
    fs.release();
    std::cout << "done." << std::endl;
    return true;
}

static bool readSomeData(const char *filename)
{
    std::cout << std::endl << "Reading: " << std::endl;
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    //fs["iterationNr"] >> itNr;
    const int itNr = fs["iterationNr"];
    std::cout << itNr << std::endl;
    if (!fs.isOpened()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    const cv::FileNode strings = fs["strings"];
    if (strings.type() != cv::FileNode::SEQ) {
        std::cerr << "strings is not a sequence! FAIL" << std::endl;
        return false;
    }

    const cv::FileNodeIterator it_end = strings.end();
    for (cv::FileNodeIterator it = strings.begin(); it != it_end; ++it) {
        const std::string &s = *it;
        // std::cout << (std::string)*it << std::endl;
        std::cout << "\"" << s << "\"" << std::endl;
    }

    const cv::FileNode mapping = fs["mapping"];
    std::cout << "Two  " << (int)(mapping["Two"]) << "; ";
    std::cout << "One  " << (int)(mapping["One"]) << std::endl << std::endl;


    SomeData someData;
    cv::Mat ucharEye, doubleZeros;

    fs["ucharEye"] >> ucharEye;
    fs["doubleZeros"] >> doubleZeros;
    fs["someData"] >> someData;

    std::cout << std::endl
              << "ucharEye = "    << std::endl << ucharEye    << std::endl
              << "doubleZeros = " << std::endl << doubleZeros << std::endl
              << std::endl
              << "someData: " << someData << std::endl
              << std::endl;

    std::cout << "Read NonExisting resulting in default data." << std::endl;
    fs["NonExisting"] >> someData;
    std::cout << "someData: " << someData << std::endl;
    return true;
}

int main(int ac, char *av[])
{
    const bool ok
        =  ac == 2
        && writeSomeData(av[1])
        && readSomeData(av[1]);
    if (ok) {
        std::cout << std::endl
                  << "Tip: Open " << av[0]
                  << " with a text editor to see the serialized data."
                  << std::endl;
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
