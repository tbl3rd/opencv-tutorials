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
    fs << "someInteger" << 100
       << "stringSequence" << "[" << "image.jpg" << "wild" << "lena.jpg" << "]"
       << "stringToIntMap" << "{" << "One" << 1 << "Two" << 2 << "}"
       << "ucharEye" << ucharEye
       << "doubleZeros" << doubleZeros
       << "someData" << someData;
    fs.release();
    std::cout << "done." << std::endl;
    return true;
}

static bool readSomeData(const char *filename)
{
    std::cout << std::endl << "Reading: " << std::endl;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    const cv::FileNode fnSomeInteger = fs["someInteger"];
    if (fnSomeInteger.type() == cv::FileNode::INT) {
        const int someInteger = fnSomeInteger;
        std::cout << "someInteger" << "=" << someInteger << std::endl;
    }

    const cv::FileNode fnStringSequence = fs["stringSequence"];
    if (fnStringSequence.type() == cv::FileNode::SEQ) {
        // std::vector<std::string> stringSequence = fnStringSequence;
        const cv::FileNodeIterator pEnd = fnStringSequence.end();
        cv::FileNodeIterator p = fnStringSequence.begin();
        const char *separator = "stringSequence" "=" "[";
        for (; p != pEnd; ++p) {
            const std::string &s = *p;
            std::cout << separator << "\"" << s << "\"";
            separator = " ";
        }
        std::cout << "]" << std::endl;
        // for (std::string s: stringSequence) {
        //     std::cout << "\"" << s << "\"" << std::endl;
        // }
    } else {
        std::cerr << "stringSequence is not a sequence!" << std::endl;
        return false;
    }

    const cv::FileNode stringToIntMap = fs["stringToIntMap"];
    if (stringToIntMap.type() == cv::FileNode::MAP) {
        std::cout << "Two " << (int)(stringToIntMap["Two"]) << "; ";
        std::cout << "One " << (int)(stringToIntMap["One"]) << std::endl
                  << std::endl;
    } else {
        std::cerr << "stringToIntMap is not a map!" << std::endl;
        return false;
    }

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
