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
    out << "{" << "aString" << " = " << m.aString << ", "
        << ""  << "aDouble" << " = " << m.aDouble << ", "
        << ""  << "anInt"   << " = " << m.anInt   << "}";
    return out;
}

int main(int ac, char *av[])
{
    if (ac != 2) {
        showUsage(av[0]);
        return 1;
    }

    std::string filename = av[1];

    { //write
        cv::Mat R = cv::Mat_<uchar>::eye(3, 3);
        cv:: Mat T = cv::Mat_<double>::zeros(3, 1);
        SomeData m(1);
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        fs << "iterationNr" << 100
           << "strings"
           << "[" << "image1.jpg" << "Awesomeness" << "baboon.jpg" << "]"
           << "Mapping" << "{" << "One" << 1 << "Two" << 2 << "}"
           << "R" << R << "T" << T
           << "SomeData" << m;
        fs.release();
        std::cout << "Write Done." << std::endl;
    }

    {//read
        std::cout << std::endl << "Reading: " << std::endl;
        cv::FileStorage fs;
        fs.open(filename, cv::FileStorage::READ);

        //fs["iterationNr"] >> itNr;
        const int itNr = fs["iterationNr"];
        std::cout << itNr;
        if (!fs.isOpened()) {
            std::cerr << "Failed to open " << filename << std::endl;
            showUsage(av[0]);
            return 1;
        }

        cv::FileNode n = fs["strings"];
        if (n.type() != cv::FileNode::SEQ) {
            std::cerr << "strings is not a sequence! FAIL" << std::endl;
            return 1;
        }

        cv::FileNodeIterator const it_end = n.end();
        for (cv::FileNodeIterator it = n.begin(); it != it_end; ++it) {
            std::cout << (std::string)*it << std::endl;
        }

        n = fs["Mapping"];
        std::cout << "Two  " << (int)(n["Two"]) << "; ";
        std::cout << "One  " << (int)(n["One"]) << std::endl << std::endl;


        SomeData m;
        cv::Mat R, T;

        fs["R"] >> R;
        fs["T"] >> T;
        fs["SomeData"] >> m;

        std::cout << std::endl
                  << "R = " << R << std::endl
                  << "T = " << T << std::endl
                  << std::endl
                  << "SomeData = " << std::endl << m << std::endl
                  << std::endl;

        std::cout << "Attempt to read NonExisting "
                  << "(should initialize the data structure with its default).";
        fs["NonExisting"] >> m;
        std::cout << std::endl << "NonExisting = "
                  << std::endl << m << std::endl;
    }

    std::cout << std::endl
              << "Tip: Open up " << filename
              << " with a text editor to see the serialized data." << std::endl;

    return 0;
}
