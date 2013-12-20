#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>


static void showUsage(const char *av0)
{
    static const char faces[]
        = "../resources/haarcascade_frontalface_alt.xml";
    static const char eyes[]
        = "../resources/haarcascade_eye_tree_eyeglasses.xml";
    std::cerr << av0 << ": Use Haar cascade classifier to find faces."
              << std::endl
              << "Usage: " << av0 << " <camera> <faces> <eyes>" << std::endl
              << std::endl
              << "Where: <camera> is an integer camera number." << std::endl
              << "       <faces> is Haar training data for faces." << std::endl
              << "       <eyes>  is Haar training data for eyes." << std::endl
              << std::endl
              << "Example: " << av0 << " 0 " << faces << " " << eyes
              << std::endl << std::endl;
}

// Return an equalized grayscale copy of image.
//
static cv::Mat grayScale(const cv::Mat &image) {
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_RGB2GRAY);
    cv::equalizeHist(result, result);
    return result;
}

static std::vector<cv::Rect> detectCascade(cv::CascadeClassifier &classifier,
                                           const cv::Mat &gray)
{
    static double scaleFactor = 1.1;
    static const int minNeighbors = 2;
    static const cv::Size minSize(30, 30);
    static const cv::Size maxSize;
    std::vector<cv::Rect> result;
    classifier.detectMultiScale(gray, result, scaleFactor, minNeighbors,
                                cv::CASCADE_SCALE_IMAGE, minSize, maxSize);
    return result;
}

static void drawFace(cv::Mat &frame, const cv::Rect &face,
                     const std::vector<cv::Rect> &eyes)
{
    static const cv::Scalar faceColor(255, 0, 255);
    static const cv::Scalar eyesColor(255, 0,   0);
    static const double angle = 0.0;
    static const double beginAngle = 0.0;
    static const double endAngle = 360.0;
    static const int thickness = 4;
    static const int lineKind = 8;
    static const int shift = 0;
    const cv::Size axes(face.width * 0.5, face.height * 0.5);
    const cv::Point center(face.x + axes.width, face.y + axes.height);
    cv::ellipse(frame, center, axes, angle, beginAngle, endAngle, faceColor,
                thickness, lineKind, shift);
    for (size_t j = 0; j < eyes.size(); ++j) {
        const cv::Rect &eye = eyes[j];
        const cv::Point center(face.x + eye.x + eye.width   * 0.5,
                               face.y + eye.y + eye.height  * 0.5);
        const int radius = cvRound((eye.width + eye.height) * 0.25);
        cv::circle(frame, center, radius, eyesColor,
                   thickness, lineKind, shift);
    }

}

static void displayFace(cv::Mat &frame,
                        cv::CascadeClassifier &faceHaar,
                        cv::CascadeClassifier &eyesHaar)

{
    const cv::Mat gray = grayScale(frame);
    const std::vector<cv::Rect> faces = detectCascade(faceHaar, gray);
    for (size_t i = 0; i < faces.size(); ++i) {
        const cv::Mat faceROI = gray(faces[i]);
        const std::vector<cv::Rect> eyes = detectCascade(eyesHaar, faceROI);
        drawFace(frame, faces[i], eyes);
    }
    cv::imshow("Capture - Face detection", frame);
}


// Just cv::VideoCapture extended for convenience.
//
struct CvVideoCapture: cv::VideoCapture {
    double framesPerSecond() {
        const double fps = this->get(cv::CAP_PROP_FPS);
        return fps ? fps : 30.0;        // for MacBook iSight camera
    }
    int fourCcCodec() {
        return this->get(cv::CAP_PROP_FOURCC);
    }
    const char *fourCcCodecString() {
        static int code = 0;
        static char result[5] = "";
        if (code == 0) {
            code = this->fourCcCodec();
            result[0] = ((code & 0x000000ff) >>  0);
            result[1] = ((code & 0x0000ff00) >>  8);
            result[2] = ((code & 0x00ff0000) >> 16);
            result[3] = ((code & 0xff000000) >> 24);
            result[4] = ""[0];
        }
        return result;
    }
    int frameCount() {
        return this->get(cv::CAP_PROP_FRAME_COUNT);
    }
    cv::Size frameSize() {
        const int w = this->get(cv::CAP_PROP_FRAME_WIDTH);
        const int h = this->get(cv::CAP_PROP_FRAME_HEIGHT);
        const cv::Size result(w, h);
        return result;
    }
    CvVideoCapture(const std::string &fileName): VideoCapture(fileName) {}
    CvVideoCapture(int n): VideoCapture(n) {}
    CvVideoCapture(): VideoCapture() {}
};


int main(int ac, const char *av[])
{
    if (ac == 4) {
        int cameraId = 0;
        std::istringstream iss(av[1]); iss >> cameraId;
        cv::CascadeClassifier faceHaar(av[2]);
        cv::CascadeClassifier eyesHaar(av[3]);
        std::cout << av[0] << ": camera ID " << cameraId << std::endl
                  << av[0] << ": Face data from " << av[2] << std::endl
                  << av[0] << ": Eyes data from " << av[3] << std::endl;
        if (!faceHaar.empty() && ! eyesHaar.empty()) {
            CvVideoCapture camera(cameraId);
            std::cout << std::endl << av[0] << ": Press any key to quit."
                      << std::endl << std::endl;
            const int msPerFrame = 1000.0 / camera.framesPerSecond();
            while (true) {
                cv::Mat frame; camera >> frame;
                if (!frame.empty()) {
                    displayFace(frame, faceHaar, eyesHaar);
                }
                const int c = cv::waitKey(msPerFrame);
                if (c != -1) break;
            }
            return 0;
        }
    }
    showUsage(av[0]);
    return 1;
}
