#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>


// Stringify identifier X to show its name.
//
#define S(X) #X

// Further stringify identifier X to show it as a C string literal.
//
#define Q(X) "\"" S(X) "\""


static void showUsage(const char *av0)
{
    std::cerr
        << std::endl
        << av0 << ": Demonstrate serializing data to and from files."
        << std::endl << std::endl
        << "Usage: " <<  av0 << " <file><ext>" << std::endl
        << std::endl
        << "Where: <file><ext> is the name of a file to read and write."
        << std::endl
        << "       The <ext> extension may be: '.xml' or '.yaml'" << std::endl
        << "       to serialize data as XML or as YAML, respectively."
        << "       The default is YAML if <ext> neither '.xml' nor '.yaml'."
        << std::endl << std::endl
        << "       A '.gz' suffix designates compression such that:"
        << std::endl
        << "           <file>.xml.gz  means use gzipped XML." << std::endl
        << "           <file>.yaml.gz means use gzipped YAML." << std::endl
        << "           '<file>.gz' is equivalent to '<file>.yaml.gz'."
        << std::endl
        << std::endl
        << "Example: " << av0 << " somedata.xml.gz" << std::endl;
}


// A class of various data members that has write() and read() functions
// sufficient for cv::FileStorage serialization.
//
class SomeData
{
    int anInt;
    double aDouble;
    std::string aString;

    // Write value with name on fs as required by cv::FileStorage.
    //
    // Use a sequence to tag the value with the class name for error
    // checking and map the names of the data members to their values.
    //
    friend void write(cv::FileStorage &fs,
                      const std::string &nameForDebuggingGuessingMaybe,
                      const SomeData &value)
    {
        std::cerr << "(" << nameForDebuggingGuessingMaybe << ")";
        fs << "[" << S(SomeData)
           <<   "{"
           <<      S(anInt)   << value.anInt
           <<      S(aDouble) << value.aDouble
           <<      S(aString) << value.aString
           <<   "}"
           << "]";
    }

    // Use some care to read value from node using defaultValue as required
    // by cv::FileStorage.
    //
    friend void read(const cv::FileNode &node, SomeData &value,
                     const SomeData &defaultValue = SomeData())
    {
        value = defaultValue;
        bool ok
            =  node.isSeq() && 2 == node.size()
            && S(SomeData) == std::string(node[0])
            && node[1].isMap() && 3 == node[1].size();
        if (ok) {
            const cv::FileNode &fn = node[1];
            ok =   fn[S(anInt)].isInt()
                && fn[S(aDouble)].isReal()
                && fn[S(aString)].isString();
            if (ok) {
                fn[S(anInt)]   >> value.anInt;
                fn[S(aDouble)] >> value.aDouble;
                fn[S(aString)] >> value.aString;
            }
        }
    }

    // Pretty-print x on os for people.
    //
    friend std::ostream &operator<<(std::ostream &os, const SomeData &x)
    {
        os << "[" << Q(SomeData) << " "
           <<   "{"
           <<     Q(anInt)   << " "         << x.anInt   << " " 
           <<     Q(aDouble) << " "         << x.aDouble << " " 
           <<     Q(aString) << " " << "\"" << x.aString << "\""
           <<   "}"
           << "]";
        return os;
    }

public:

    SomeData(): anInt(1), aDouble(1.1), aString("default ctor") {}
    SomeData(int): anInt(97), aDouble(CV_PI), aString("mydata1234") {}
};


// Write someInteger, stringSequence, stringToIntMap, ucharEye,
// doubleZeros, and someData to filename.  Return true.
//
static bool writeSomeStuff(const char *filename)
{
    static const cv::Mat ucharEye = cv::Mat_<uchar>::eye(3, 3);
    static const cv::Mat doubleZeros = cv::Mat_<double>::zeros(3, 1);
    static const SomeData someData(1);
    std::cout << std::endl << "Writing " << filename << " ... ";
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << S(someInteger) << 100
       << S(stringSequence) << "[" << "image.jpg" << "wild" << "lena.jpg" << "]"
       << S(stringToIntMap) << "{" << "One" << 1 << "Two" << 2 << "}"
       << S(ucharEye) << ucharEye
       << S(doubleZeros) << doubleZeros
       << S(someData) << someData;
    std::cout << " ... done." << std::endl;
    return true;
}

// Read someInteger from fs and show it as a map literal.
//
static bool readSomeInteger(const cv::FileStorage &fs)
{
    const cv::FileNode &fnSomeInteger = fs[S(someInteger)];
    if (fnSomeInteger.isInt()) {
        const int someInteger = fnSomeInteger;
        std::cout << "{" << Q(someInteger)
                  << " " <<   someInteger << "}" << std::endl;
        return true;
    }
    std::cerr << S(someInteger) " is not an integer" << std::endl;
    return false;
}

// Read stringSequence from fs and show it as a map literal.
//
static bool readStringSequence(const cv::FileStorage &fs)
{
    const cv::FileNode &fnStringSequence = fs[S(stringSequence)];
    if (fnStringSequence.isSeq()) {
        const cv::FileNodeIterator pEnd = fnStringSequence.end();
        cv::FileNodeIterator p = fnStringSequence.begin();
        const char *separator = "{" Q(stringSequence) " " "[";
        for (; p != pEnd; ++p) {
            if ((*p).type() == cv::FileNode::STRING) {
                const std::string s(*p);
                std::cout << separator << "\"" << s << "\"";
                separator = " ";
            } else {
                std::cerr << S(stringSequence) " element is not a string!"
                          << std::endl;
                return false;
            }
        }
        std::cout << "]" << "}" << std::endl;
        return true;
    }
    std::cerr << S(stringSequence) " is not a sequence!" << std::endl;
    return false;
}

// Read stringToIntMap from fs and show it as a map literal.
//
static bool readStringToIntMap(const cv::FileStorage &fs)
{
    const cv::FileNode &fnStringToIntMap = fs[S(stringToIntMap)];
    if (fnStringToIntMap.isMap()) {
        const cv::FileNodeIterator pEnd = fnStringToIntMap.end();
        cv::FileNodeIterator p = fnStringToIntMap.begin();
        const char *separator = "{" Q(stringToIntMap) " " "{";
        for (; p != pEnd; ++p) {
            if ((*p).isNamed()) {
                const std::string n((*p).name());
                const int v = *p;
                std::cout << separator << "\"" << n << "\"" << " " << v;
                separator = " ";
            } else {
                std::cerr << S(stringToIntMap) " node is not named!"
                          << std::endl;
                return false;
            }
        }
        std::cout << "}" << "}" << std::endl;
        return true;
    }
    std::cerr << S(stringToIntMap) " is not a map!" << std::endl;
    return false;
}

// Without checking for errors, read ucharEye, doubleZeros, and someData
// from fs and show them as map literals.
//
static bool readMatAndSomeData(const cv::FileStorage &fs)
{
    cv::Mat ucharEye;
    cv::Mat doubleZeros;
    SomeData someData;
    fs[S(ucharEye)]    >> ucharEye;
    fs[S(doubleZeros)] >> doubleZeros;
    fs[S(someData)]    >> someData;
    std::cout << std::endl
              << "{" Q(ucharEye) " " << std::endl
              << ucharEye << std::endl << "}" << std::endl << std::endl;
    std::cout << "{" Q(doubleZeros) " " << doubleZeros << "}"
              << std::endl << std::endl;
    std::cout << "{" Q(someData) " " << someData << "}" << std::endl
              << std::endl;
    return true;
}

// Look up the bogus name "no thing" in fs and show the results for various
// data types.  Return true.
//
static bool readNothing(const cv::FileStorage &fs)
{
    std::cout << "Read \"no thing\" into various types." << std::endl;
    int noInt;
    double noDouble;
    std::string noString;
    cv::Mat noMat;
    SomeData noSomeData;
    fs["no thing"] >> noInt;
    fs["no thing"] >> noDouble;
    fs["no thing"] >> noString;
    fs["no thing"] >> noMat;
    fs["no thing"] >> noSomeData;
    std::cout << S(noInt)      ": " <<        noInt             << std::endl;
    std::cout << S(noDouble)   ": " <<        noDouble          << std::endl;
    std::cout << S(noString)   ": " << "'" << noString   << "'" << std::endl;
    std::cout << S(noMat)      ": " <<        noMat             << std::endl;
    std::cout << S(noSomeData) ": " <<        noSomeData        << std::endl;
    return true;
}

// Read some stuff from filename.  Return true on success.  Return false on
// failure.
//
static bool readSomeStuff(const char *filename)
{
    std::cout << "Reading " << filename << " back." << std::endl << std::endl;
    const cv::FileStorage fs(filename, cv::FileStorage::READ);
    bool ok = fs.isOpened();
    if (!ok) std::cerr << "Failed to open " << filename << std::endl;
    ok = ok
        && readSomeInteger(fs)
        && readStringSequence(fs)
        && readStringToIntMap(fs)
        && readMatAndSomeData(fs)
        && readNothing(fs);
    return ok;
}

int main(int ac, char *av[])
{
    const bool ok
        =  ac == 2
        && writeSomeStuff(av[1])
        && readSomeStuff(av[1]);
    if (ok) {
        std::cerr << std::endl
                  << "Tip: Open " << av[1]
                  << " with a text editor to see the serialized data."
                  << std::endl << std::endl;
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
